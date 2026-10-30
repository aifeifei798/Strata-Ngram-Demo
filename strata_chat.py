#!/usr/bin/env python
"""Strata 风格的 LFM3-VL chat:模型权重全部驻留 GPU 显存,自建的 n-gram 查找表
留在 SSD 上通过 mmap 按需读取(热行由 OS 页缓存托管),用它做 speculative
decoding 的 draft 源,由主模型整批验证 —— 输出与不开 draft 完全一致。

对应 Strata 的三级分层:GPU=模型与 KV cache;RAM=页缓存中的表热行;SSD=表本体。
--table-io mmap|ram 对应 Strata 的 --ple-io mmap|ram。

用法:
    # 1. 建表(语料换成你手头任何文本都行;表建议放 NVMe/SSD 上)
    python build_ngram_table.py --corpus 语料.jsonl --out strata_ngram.bin

    # 2. 聊天(默认纯文本多轮;看图加 --image 路径)
    python strata_chat.py --interactive
    python strata_chat.py --image photo.jpg --text "详细描述这张照片"

    # 3. 实验模式
    python strata_chat.py --check    # greedy 自检:strata 与 baseline 输出逐 token 一致
    python strata_chat.py --bench    # baseline / pld / strata 三模式测速
    python strata_chat.py --bench --cold-table   # 冷表演示:生成前丢页缓存,观察真实回盘
    python strata_chat.py --bench --greedy --image "" \
        --text "请一字不差地重复下面方括号里的内容:[...一段话...]"   # draft 收益最大的场景
"""

import argparse
import mmap
import os
import struct
import time
from threading import Thread

import numpy as np
import torch
import transformers.generation.utils as gen_utils
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.generation.candidate_generator import PromptLookupCandidateGenerator

DEFAULT_MODEL = "../LFM3-VL-2B"  # 本地模型目录,或 HF hub id(如 LiquidAI/LFM3-VL-2B)
DEFAULT_IMAGE = ""  # 想让首轮看图,--image 传本地图片路径
DEFAULT_TEXT = "你好,用中文简单介绍一下你自己,两句话就行"
TABLE_PATH = "strata_ngram.bin"
TRI_TABLE_PATH = "strata_tri_ngram.bin"
HEADER_SIZE = 24  # struct("<8sIIQ") = 8s magic + I + I + Q

NEW_STATS = lambda: {"prompt_rounds": 0, "table_rounds": 0, "table_refills": 0,
                     "drafted_rounds": 0, "draft_tokens": 0,
                     "verified_tokens": 0, "verify_rounds": 0,
                     "tri_steps": 0, "bi_steps": 0,
                     "rows_read": 0, "query_time": 0.0}


# ---------------------------------------------------------------------------
# OS 层 IO 统计:把"表在 SSD 上、由页缓存分层"变成可测量的数字
# ---------------------------------------------------------------------------
def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def proc_io_snapshot() -> dict:
    """进程级 IO 计数:read_bytes = 块层真实从存储读的字节;majflt = 真回盘缺页。"""
    snap = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                key, _, val = line.rstrip().partition(": ")
                if key in ("rchar", "read_bytes"):
                    snap[key] = int(val)
    except OSError:
        pass
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rpartition(")")[2].split()
        snap["minflt"], snap["majflt"] = int(fields[0]), int(fields[1])
    except (OSError, IndexError, ValueError):
        pass
    return snap


def file_rss_bytes(path: str) -> int:
    """该文件当前驻留在页缓存中的字节数(从 /proc/self/smaps 读其映射的 Rss)。"""
    target = os.path.realpath(path)
    total = 0
    try:
        with open("/proc/self/smaps") as f:
            in_mapping = False
            for line in f:
                head = line.split("-", 1)
                try:
                    int(head[0], 16)
                    is_vma = True  # "addr-addr perm offset dev inode path" 头行
                except ValueError:
                    is_vma = False
                if is_vma:
                    in_mapping = line.rstrip().endswith(target)
                elif in_mapping and line.startswith("Rss:"):
                    total += int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return total


# ---------------------------------------------------------------------------
# SSD 上的 n-gram 表:mmap 映射,不占显存、零拷贝,按行按需读
# ---------------------------------------------------------------------------
class StrataTable:
    label = "bigram 表"

    def __init__(self, path: str, io: str = "mmap"):
        self.rows_read = 0
        self.query_time = 0.0
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            magic, vocab, topk, data_offset = struct.unpack("<8sIIQ", f.read(HEADER_SIZE))
        if magic != b"STRNTBL1":
            raise ValueError(f"{path} 不是有效的 n-gram 表(magic={magic!r})")
        self.vocab, self.topk, self.path = vocab, topk, path
        self.mode, self.size_bytes = io, size

        if io == "ram":
            # --ple-io ram:整表读进内存,查询走 RAM,不吃页缺失
            with open(path, "rb") as f:
                f.seek(HEADER_SIZE)
                buf = f.read(size - HEADER_SIZE)
            self._file, self._mm = None, None
            self.offsets = np.frombuffer(buf, dtype="<u8", count=vocab + 1)
            self.rows = np.frombuffer(buf, dtype="<u4", offset=(vocab + 1) * 8)
        else:
            # --ple-io mmap:文件映射进地址空间,按需调页;
            # 内存吃紧时内核自动逐出冷页,表本体永远"住在 SSD 上"
            self._file = open(path, "rb")
            self._mm = mmap.mmap(self._file.fileno(), 0, prot=mmap.PROT_READ)
            self.offsets = np.frombuffer(self._mm, dtype="<u8", count=vocab + 1, offset=HEADER_SIZE)
            self.rows = np.frombuffer(self._mm, dtype="<u4", offset=data_offset)

    def next_tokens(self, prev_token: int, k: int = 1) -> list[int]:
        """读一行:行号 = 前驱 token id,返回该行频次最高的 k 个后继。"""
        t0 = time.perf_counter()
        s, e = int(self.offsets[prev_token]), int(self.offsets[prev_token + 1])
        self.rows_read += 1
        self.query_time += time.perf_counter() - t0
        if e <= s:
            return []
        return self.rows[s:e][:k].tolist()

    def drop_cache(self):
        """posix_fadvise DONTNEED:丢弃表的页缓存,模拟 Strata --ple-io direct 的最冷情形。
        只读映射的页永远是 clean 页,必然可丢弃;之后访问会 major fault 真回盘。"""
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def resident_bytes(self) -> int:
        return 0 if self.mode == "ram" else file_rss_bytes(self.path)


class TriTable:
    """trigram(二阶)表:key = (前两 token) 的组合,按 key 排序存储,查询 = 二分。
    组合空间是词表^2,没法像 bigram 那样按行号直接索引;每次查询读 log2(n) 个页,
    正好对应 Strata 的 "read a few rows per token"。"""
    label = "trigram 表"

    def __init__(self, path: str):
        self.rows_read = 0
        self.query_time = 0.0
        with open(path, "rb") as f:
            magic, vocab, n, data_offset = struct.unpack("<8sIIQ", f.read(HEADER_SIZE))
        if magic != b"STRTBL21":
            raise ValueError(f"{path} 不是有效的 trigram 表(magic={magic!r})")
        self.vocab, self.n, self.path = vocab, n, path
        self.mode, self.size_bytes = "mmap", os.path.getsize(path)
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, prot=mmap.PROT_READ)
        self.keys = np.frombuffer(self._mm, dtype="<u8", count=n, offset=data_offset)
        self.nexts = np.frombuffer(self._mm, dtype="<u4", count=n, offset=data_offset + n * 8)

    def next_token(self, t2: int, t1: int) -> int | None:
        """二分查 (t2,t1) 的 top-1 后继;未登录返回 None。
        标量必须转 np.uint64:NEP 50 下 uint64 数组与 Python int 比较会触发
        类型提升,searchsorted 退化成全表扫描(实测慢 330 倍)。"""
        t0 = time.perf_counter()
        key = t2 * self.vocab + t1
        idx = int(np.searchsorted(self.keys, np.uint64(key)))
        hit = idx < self.n and int(self.keys[idx]) == key
        self.rows_read += 1
        self.query_time += time.perf_counter() - t0
        return int(self.nexts[idx]) if hit else None

    def drop_cache(self):
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def resident_bytes(self) -> int:
        return file_rss_bytes(self.path)


# ---------------------------------------------------------------------------
# draft 生成器:prompt-lookup 优先(上下文里找重复段),不够长用 SSD 表链式补足
# ---------------------------------------------------------------------------
class StrataCandidateGenerator(PromptLookupCandidateGenerator):
    def __init__(self, *args, table: StrataTable, stats: dict, tri: TriTable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.table = table
        self.tri = tri
        self.stats = stats

    def _sync_stats(self):
        self.stats["rows_read"] = self.table.rows_read + (self.tri.rows_read if self.tri else 0)
        self.stats["query_time"] = self.table.query_time + (self.tri.query_time if self.tri else 0.0)

    def _chain_from_table(self, prev2: int, prev1: int, max_len: int) -> list[int]:
        """链式 draft:每步优先查 trigram(看前两个 token),查不到退回 bigram top-1。"""
        draft: list[int] = []
        for _ in range(max_len):
            nxt = self.tri.next_token(prev2, prev1) if self.tri is not None else None
            if nxt is not None:
                self.stats["tri_steps"] += 1
            else:
                got = self.table.next_tokens(prev1, 1)
                if not got:
                    break
                nxt = got[0]
                self.stats["bi_steps"] += 1
            draft.append(nxt)
            prev2, prev1 = prev1, nxt
        if self.eos_token_id is not None and draft:
            eos = self.eos_token_id.detach().flatten().to("cpu", torch.long)
            hit = torch.nonzero(torch.isin(torch.tensor(draft, dtype=torch.long), eos))
            if hit.numel() > 0:
                draft = draft[: hit[0].item()]
        return draft

    def get_candidates(self, input_ids, **kwargs):
        out, scores = super().get_candidates(input_ids, **kwargs)  # prompt-lookup 优先
        t2 = int(input_ids[0, -2]) if input_ids.shape[1] >= 2 else int(input_ids[0, -1])
        t1 = int(input_ids[0, -1])
        if out.shape[1] > input_ids.shape[1]:
            self.stats["prompt_rounds"] += 1
            # 上下文里抄到的续写不足额时,用表从段尾接着往下补
            shortfall = self.num_output_tokens - (out.shape[1] - input_ids.shape[1])
            if shortfall > 0:
                extra = self._chain_from_table(int(out[0, -2]) if out.shape[1] >= 2 else t1,
                                               int(out[0, -1]), shortfall)
                if extra:
                    self.stats["table_refills"] += 1
                    extra_t = torch.tensor([extra], device=input_ids.device, dtype=input_ids.dtype)
                    out = torch.cat((out, extra_t), dim=1)
        else:
            draft = self._chain_from_table(t2, t1, self.num_output_tokens)
            self.stats["table_rounds"] += 1
            if draft:
                draft_t = torch.tensor([draft], device=input_ids.device, dtype=input_ids.dtype)
                out = torch.cat((input_ids, draft_t), dim=1)
        if out.shape[1] > input_ids.shape[1]:
            self.stats["drafted_rounds"] += 1
            self.stats["draft_tokens"] += out.shape[1] - input_ids.shape[1]
        self._sync_stats()
        return out, scores

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        self.stats["verified_tokens"] += num_matches
        self.stats["verify_rounds"] += 1
        self._sync_stats()
        return super().update_candidate_strategy(input_ids, scores, num_matches)


# ---------------------------------------------------------------------------
def enable_strata(table: StrataTable, stats: dict, tri: TriTable | None = None):
    """generate() 每次都会按 generation_config 实例化 PromptLookupCandidateGenerator,
    把那个类指向我们的实现即完成注入,验证逻辑全部复用 transformers 的。"""
    def factory(*args, **kwargs):
        return StrataCandidateGenerator(*args, table=table, stats=stats, tri=tri, **kwargs)

    gen_utils.PromptLookupCandidateGenerator = factory


def disable_strata():
    gen_utils.PromptLookupCandidateGenerator = PromptLookupCandidateGenerator


def load_model(model_id):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    return model


def run_generation(model, processor, messages, gen_kwargs, stats, tables=None, cold_table=False):
    if tables and cold_table:
        for t in tables:
            t.drop_cache()  # 每轮都从全冷开始:页缓存里一行都没有,全部真回盘
    io_before = proc_io_snapshot()

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _generate():
        try:
            model.generate(**inputs, streamer=streamer, **gen_kwargs)
        except Exception as exc:
            print(f"\n[generate 线程异常] {exc!r}", flush=True)
            streamer.end()  # 否则主线程会在 streamer 上永远等下去

    thread = Thread(target=_generate)
    t0 = time.perf_counter()
    thread.start()

    chunks = []
    first_chunk_t = None
    for chunk in streamer:
        if chunk and first_chunk_t is None:
            first_chunk_t = time.perf_counter()
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    thread.join()
    elapsed = time.perf_counter() - t0
    decode_t = time.perf_counter() - first_chunk_t if first_chunk_t else elapsed
    print()

    reply = "".join(chunks)
    n_tok = len(processor.tokenizer(reply, add_special_tokens=False)["input_ids"])
    tok_s = n_tok / decode_t if decode_t > 0 else 0.0
    print(f"[生成] {n_tok} tokens | 总耗时 {elapsed:.1f}s | 生成期 {tok_s:.1f} tok/s")
    if stats["drafted_rounds"]:
        drafted, accepted = stats["draft_tokens"], stats["verified_tokens"]
        rate = accepted / drafted if drafted else 0.0
        rows_s = stats["rows_read"] / stats["query_time"] if stats["query_time"] > 0 else 0
        print(
            f"[draft] 参与 draft 的验证轮 {stats['drafted_rounds']}"
            f"(prompt 查找 {stats['prompt_rounds']} / 表链 {stats['table_rounds']}"
            f" / 表补足 {stats['table_refills']};链式查询 tri {stats['tri_steps']} 步"
            f" / bi 回退 {stats['bi_steps']} 步),"
            f" 提出草案 {drafted} 个,采纳 {accepted} 个(接受率 {rate:.0%})"
        )
        print(f"[表] 行读取 {stats['rows_read']:,} | {rows_s:,.0f} rows/s | 显存占用:表不占 VRAM")

    if tables:
        io_after = proc_io_snapshot()
        disk_read = io_after.get("read_bytes", 0) - io_before.get("read_bytes", 0)
        majflt = io_after.get("majflt", 0) - io_before.get("majflt", 0)
        for t in tables:
            if t.mode == "mmap":
                resident = t.resident_bytes()
                pct = resident / t.size_bytes * 100 if t.size_bytes else 0.0
                cache_str = f" | 页缓存驻留 {fmt_bytes(resident)}/{fmt_bytes(t.size_bytes)} ({pct:.0f}%)"
            else:
                cache_str = " | 读表方式 ram(整表驻内存)"
            print(f"[IO] {t.label} {fmt_bytes(t.size_bytes)}{cache_str}"
                  f" | 本次生成真实磁盘读 {fmt_bytes(disk_read)} | major fault {majflt}")
    return reply, n_tok, tok_s


def print_vram():
    print(f"[显存] 峰值 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB(查找表不占其中)")


def load_tables(args):
    """加载 bigram 表(必需)+ trigram 表(可选;文件不存在则优雅降级只用 bigram)。"""
    table = StrataTable(args.table, io=args.table_io)
    tri = None
    if args.tri_table and os.path.exists(args.tri_table):
        tri = TriTable(args.tri_table)
        print(f"[表] {tri.label}已加载: {args.tri_table}({fmt_bytes(tri.size_bytes)},{tri.n:,} 条)")
    else:
        print(f"[表] 未找到 trigram 表({args.tri_table}),draft 链只用 bigram;"
              f"用 build_ngram_table.py --trigram 生成后 --tri-table 指过来")
    return table, tri


def main():
    ap = argparse.ArgumentParser(description="Strata 风格 LFM3-VL chat")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型路径或 HF hub id")
    ap.add_argument("--table", default=TABLE_PATH, help="bigram 表文件(建议放 SSD)")
    ap.add_argument("--tri-table", default=TRI_TABLE_PATH,
                    help="trigram 表文件(可选;文件不存在则只用 bigram)")
    ap.add_argument("--table-io", choices=["mmap", "ram"], default="mmap",
                    help="mmap=按需调页(默认);ram=整表进内存")
    ap.add_argument("--draft-k", type=int, default=8, help="每轮最多 draft 的 token 数")
    ap.add_argument("--mode", choices=["strata", "pld", "baseline"], default="strata",
                    help="strata=prompt查找+SSD表;pld=仅原版 prompt lookup;baseline=不开 draft")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="首轮对话的图片(传空字符串则纯文本)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="首轮提问")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--interactive", action="store_true", help="多轮交互 chat")
    ap.add_argument("--greedy", action="store_true", help="greedy 采样(repetition penalty 会压制重复 prompt 里的 token,n-gram draft 建议关掉惩罚用 greedy 看效果)")
    ap.add_argument("--cold-table", action="store_true",
                    help="每次生成前丢掉表的页缓存(fadvise DONTNEED),演示表完全在 SSD 上的最冷情形")
    ap.add_argument("--check", action="store_true", help="greedy 下验证 strata 输出与 baseline 完全一致")
    ap.add_argument("--bench", action="store_true", help="三种模式各跑一轮并对比 tok/s")
    args = ap.parse_args()

    model = load_model(args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print_vram()

    sample_kwargs = (dict(do_sample=False) if (args.check or args.greedy)
                     else dict(do_sample=True, temperature=0.1, top_k=50, repetition_penalty=1.1))

    def build_kwargs(extra=None):
        kw = dict(max_new_tokens=args.max_new_tokens, **sample_kwargs)
        if extra:
            kw.update(extra)
        return kw

    def make_messages(user_text, with_image):
        content = []
        if with_image:
            content.append({"type": "image", "url": args.image})
        content.append({"type": "text", "text": user_text})
        return [{"role": "user", "content": content}]

    # --- 一致性自检:assisted decoding 的保证是输出与逐 token 生成完全一致 ---
    if args.check:
        outputs = {}
        for mode in ("baseline", "strata"):
            stats = NEW_STATS()
            gen_kwargs = build_kwargs({"prompt_lookup_num_tokens": args.draft_k} if mode == "strata" else None)
            if mode == "strata":
                table, tri = load_tables(args)
                enable_strata(table, stats, tri=tri)
            print(f"\n=== check [{mode}] ===")
            tables = [t for t in (table, tri) if t] if mode == "strata" else None
            extra = {"tables": tables, "cold_table": args.cold_table} if mode == "strata" else {}
            reply, _, _ = run_generation(model, processor, make_messages(args.text, bool(args.image)), gen_kwargs, stats, **extra)
            outputs[mode] = reply
            print_vram()
            disable_strata()
        same = outputs["baseline"] == outputs["strata"]
        print(f"\n[check 结果] baseline 与 strata 输出{'一致 ✅ draft 只影响速度,不影响结果' if same else '不一致 ❌'}")
        return

    # --- benchmark:三种模式各跑一轮 ---
    if args.bench:
        rows = []
        for mode in ("baseline", "pld", "strata"):
            stats = NEW_STATS()
            gen_kwargs = build_kwargs(
                {"prompt_lookup_num_tokens": args.draft_k} if mode != "baseline" else None)
            if mode == "strata":
                table, tri = load_tables(args)
                enable_strata(table, stats, tri=tri)
            print(f"\n=== bench [{mode}] ===")
            tables = [t for t in (table, tri) if t] if mode == "strata" else None
            extra = {"tables": tables, "cold_table": args.cold_table} if mode == "strata" else {}
            _, n_tok, tok_s = run_generation(model, processor, make_messages(args.text, bool(args.image)), gen_kwargs, stats, **extra)
            rows.append((mode, n_tok, tok_s))
            print_vram()
            disable_strata()
        print("\n=== 汇总 ===")
        for mode, n_tok, tok_s in rows:
            print(f"  {mode:<9} {n_tok:>4} tokens  {tok_s:>6.1f} tok/s")
        return

    # --- 默认:单轮或交互 chat ---
    table, tri = load_tables(args) if args.mode == "strata" else (None, None)
    stats = NEW_STATS()
    if table is not None:
        enable_strata(table, stats, tri=tri)
    tables = [t for t in (table, tri) if t] if table is not None else None

    gen_kwargs = build_kwargs({"prompt_lookup_num_tokens": args.draft_k} if args.mode != "baseline" else None)

    messages = make_messages(args.text, bool(args.image))
    while True:
        print(f"\n[user] {args.text}")
        reply, _, _ = run_generation(model, processor, messages, gen_kwargs, stats, tables=tables, cold_table=args.cold_table)
        print_vram()
        if not args.interactive:
            break
        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        try:
            user_text = input("\n[user] (回车或 Ctrl-C 退出) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见")
            break
        if not user_text:
            break
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        args.text = user_text


if __name__ == "__main__":
    main()
