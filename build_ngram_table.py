#!/usr/bin/env python
"""构建 Strata 风格的 n-gram 查找表,存为可 mmap 的二进制文件。

默认(--bigram):
    header  : magic b'STRNTBL1' | uint32 vocab | uint32 topk | uint64 data_offset
    offsets : uint64[vocab+1]   行 i 的条目区间为 [offsets[i], offsets[i+1])
    rows    : uint32[...]       next token id,行内按语料频次降序
    行号就是前驱 token id,查询 = 按下标直接读一行。

--trigram(二阶表):
    header  : magic b'STRTBL21' | uint32 vocab | uint32 n | uint64 data_offset
    keys    : uint64[n]         key = t[i]*vocab + t[i+1],升序排列
    nexts   : uint32[n]         该上下文下语料中频次最高的后继(top-1)
    组合空间是词表^2,无法按行号直接索引,改为排序数组 + 二分:
    每次查询读 log2(n) 个页,正好对应 Strata 的 "read a few rows per token"。

两种表都不常驻任何显存/内存,由 OS 页缓存按热度管理,冷行留在 SSD 上。
"""

import argparse
import json
import struct
from collections.abc import Iterator

import numpy as np
from transformers import AutoConfig, AutoTokenizer

MAGIC = b"STRNTBL1"
TRI_MAGIC = b"STRTBL21"
HEADER_FMT = "<8sIIQ"  # 8 字节 magic + vocab + topk/n + 数据区偏移
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def iter_texts(paths: list[str]) -> Iterator[str]:
    """从 jsonl/text 文件里抽出所有字符串字段作为语料文本。"""
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue
                yield from _collect_strings(obj)


def _collect_strings(x) -> Iterator[str]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _collect_strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _collect_strings(v)


def tokenize_corpus(texts: Iterator[str], tokenizer, token_budget: int) -> np.ndarray:
    """把语料切成 token id 流(拼接所有文本,忽略文本边界,总量受 token_budget 限制)。"""
    parts: list[np.ndarray] = []
    total = 0
    for text in texts:
        if total >= token_budget:
            break
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        parts.append(np.asarray(ids, dtype=np.uint32))
        total += len(ids)
        if total % 2_000_000 < len(ids):
            print(f"  已切分 {total:,} tokens ...", flush=True)
    if not parts:
        raise SystemExit("语料为空:没有切出任何 token")
    return np.concatenate(parts)


def build_table(ids: np.ndarray, vocab: int, topk: int, min_count: int):
    """统计 bigram 频次,每个前驱 token 取 topk 个后继,返回 (offsets, rows)。"""
    pairs = ids[:-1].astype(np.uint64) * np.uint64(vocab) + ids[1:]
    print(f"  bigram 出现 {pairs.size:,} 次,开始统计唯一组合 ...", flush=True)
    keys, counts = np.unique(pairs, return_counts=True)
    prev = (keys // np.uint64(vocab)).astype(np.int64)
    nxt = (keys % np.uint64(vocab)).astype(np.int32)
    counts = counts.astype(np.int64)

    keep = counts >= min_count
    prev, nxt, counts = prev[keep], nxt[keep], counts[keep]
    print(f"  唯一 bigram {keys.size:,},过滤 count>={min_count} 后剩 {prev.size:,}", flush=True)

    # prev 升序 + 组内 count 降序
    order = np.lexsort((-counts, prev))
    prev, nxt = prev[order], nxt[order]
    group_start = np.searchsorted(prev, np.arange(vocab + 1))
    pos = np.arange(prev.size)
    within = pos - group_start[prev]  # 每个条目在自己组里的排名
    take = within < topk

    offsets = np.zeros(vocab + 1, dtype=np.uint64)
    group_len = group_start[1:] - group_start[:-1]
    offsets[1:] = np.minimum(group_len, topk).cumsum().astype(np.uint64)
    rows = nxt[take].astype(np.uint32)
    return offsets, rows


def build_trigram_table(ids: np.ndarray, vocab: int, min_count: int):
    """统计 trigram 频次,每个 (前两 token) 组合取频次最高的后继。
    返回按 key 升序的 (keys, nexts),查询用二分。"""
    ctx = ids[:-2].astype(np.uint64) * np.uint64(vocab) + ids[1:-1]
    nxt = ids[2:]
    print(f"  trigram 出现 {ctx.size:,} 次,排序去重(峰值内存约 3 GiB)...", flush=True)

    order = np.lexsort((nxt, ctx))  # ctx 升序,next 升序 -> (ctx,next) 相同的聚成连续段
    k, n2 = ctx[order], nxt[order]
    del ctx, nxt, order

    boundary = np.ones(k.size, dtype=bool)
    boundary[1:] = (k[1:] != k[:-1]) | (n2[1:] != n2[:-1])
    starts = np.flatnonzero(boundary)
    counts = np.diff(np.append(starts, k.size)).astype(np.int64)  # 每个 (key,next) 的出现次数
    pk, pn = k[starts], n2[starts]
    del k, n2, boundary, starts

    keep = counts >= min_count
    pk, pn, counts = pk[keep], pn[keep], counts[keep]
    print(f"  唯一 (上下文,后继) 组合 {keep.size:,},过滤 count>={min_count} 后剩 {pk.size:,}", flush=True)

    # key 升序 + 组内 count 降序,每组取第一条 = 该上下文的 top-1 后继
    order2 = np.lexsort((-counts, pk))
    pk, pn = pk[order2], pn[order2]
    first = np.ones(pk.size, dtype=bool)
    first[1:] = pk[1:] != pk[:-1]
    return pk[first].astype(np.uint64), pn[first].astype(np.uint32)


def main():
    ap = argparse.ArgumentParser(description="构建 Strata 风格 n-gram 查找表")
    ap.add_argument("--corpus", nargs="+", required=True, help="语料文件(jsonl 或纯文本)")
    ap.add_argument("--tokenizer", default="../LFM3-VL-2B")
    ap.add_argument("--out", default="strata_ngram.bin", help="表输出路径(想放 SSD 就指向那个盘)")
    ap.add_argument("--topk", type=int, default=32, help="每行保留的后继 token 数")
    ap.add_argument("--min-count", type=int, default=2, help="低于该频次的 bigram 丢弃")
    ap.add_argument("--trigram", action="store_true",
                    help="构建 trigram 表(前两 token -> top-1 后继,排序数组 + 二分查询)")
    ap.add_argument("--max-tokens", type=int, default=24_000_000, help="语料 token 预算")
    args = ap.parse_args()

    vocab = AutoConfig.from_pretrained(args.tokenizer, trust_remote_code=True).get_text_config().vocab_size
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"词表大小 {vocab:,},开始切分语料 {args.corpus}", flush=True)

    ids = tokenize_corpus(iter_texts(args.corpus), tokenizer, args.max_tokens)

    if args.trigram:
        print(f"语料共 {ids.size:,} tokens,统计 trigram ...", flush=True)
        keys, nexts = build_trigram_table(ids, vocab, args.min_count)
        del ids
        with open(args.out, "wb") as f:
            f.write(struct.pack(HEADER_FMT, TRI_MAGIC, vocab, keys.size, HEADER_SIZE))
            f.write(keys.tobytes())
            f.write(nexts.tobytes())
        print()
        print(f"trigram 表已写入 {args.out}")
        print(f"  条目(唯一上下文): {keys.size:,}")
        print(f"  文件大小: {keys.nbytes + nexts.nbytes + HEADER_SIZE:,} 字节")

        # mmap 读回自测:取表中一个真实条目,decode 展示 + 验证二分命中
        mm = np.memmap(args.out, dtype=np.uint8, mode="r")
        mkeys = np.frombuffer(mm[HEADER_SIZE:HEADER_SIZE + keys.nbytes].tobytes(), dtype=np.uint64)
        mnexts = np.frombuffer(mm[HEADER_SIZE + keys.nbytes:].tobytes(), dtype=np.uint32)
        i = keys.size // 2
        t2, t1 = divmod(int(mkeys[i]), vocab)
        pred = int(mnexts[i])
        print(f"  自测: 上下文 {tokenizer.decode([t2])!r}{tokenizer.decode([t1])!r}"
              f" 的 top-1 后继: {pred} {tokenizer.decode([pred])!r}")
        idx = int(np.searchsorted(mkeys, int(mkeys[i])))
        assert idx < keys.size and int(mkeys[idx]) == int(mkeys[i]), "二分自测失败"
        print(f"  二分自测: searchsorted 命中 idx={idx} ✅")
        return

    print(f"语料共 {ids.size:,} tokens,统计 bigram ...", flush=True)

    offsets, rows = build_table(ids, vocab, args.topk, args.min_count)
    data_offset = HEADER_SIZE + offsets.nbytes
    with open(args.out, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, vocab, args.topk, data_offset))
        f.write(offsets.tobytes())
        f.write(rows.tobytes())

    nonempty = int((offsets[1:] > offsets[:-1]).sum())
    print()
    print(f"表已写入 {args.out}")
    print(f"  行数(词表): {vocab:,},非空行: {nonempty:,} ({nonempty / vocab:.1%})")
    print(f"  总条目: {rows.size:,},平均行宽: {rows.size / max(nonempty, 1):.1f}")
    print(f"  文件大小: {offsets.nbytes + rows.nbytes + HEADER_SIZE:,} 字节")

    # 用 mmap 重新读回文件自测(顺便验证查询路径)
    mm = np.memmap(args.out, dtype=np.uint8, mode="r")
    off2 = np.frombuffer(mm[HEADER_SIZE:data_offset].tobytes(), dtype=np.uint64)
    rows2 = np.frombuffer(mm[data_offset:].tobytes(), dtype=np.uint32)
    for prev_id in (vocab // 2, 0):
        s, e = int(off2[prev_id]), int(off2[prev_id + 1])
        toks = rows2[s:e][:5]
        print(f"  行 {prev_id} 的 top-5 后继: {toks.tolist()} -> {tokenizer.decode(toks.tolist())!r}")


if __name__ == "__main__":
    main()
