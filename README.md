# Strata-Ngram-Demo

参考 [Strata](https://github.com/niko1221/strata) 的"大表放 SSD、其余放显存"思路,
给 **LFM3-VL-2B** 复刻的三层分层推理 demo:

| 层 | Strata | 本 demo |
|---|---|---|
| GPU 显存 | 热点专家权重、注意力、输出头 | 整个 LFM3-VL-2B(约 3.6 GiB)+ KV cache |
| 内存 | 全部专家权重(pinned) | 表的热行(OS 页缓存自动托管) |
| SSD | 28.8 GB n-gram 查找表 | `strata_ngram.bin`(语料越多越大) |

自建的 n-gram 查找表通过 `mmap` 映射,**不占任何显存**;行号 = 前驱 token id,
每生成一个词元按下标直接读几行(实测约 70–85 万 rows/s,热行由页缓存接管,
冷行留在盘上)。draft 交给 transformers 的 assisted generation 整批验证,
**输出与不开 draft 完全一致**(`--check` 可自证)。

draft 来源两级,对应 Strata 的 drafter 设计:

1. **prompt lookup**(不占空间):从当前上下文里找重复段直接抄;
2. **SSD 上的 n-gram 表**(持久统计):上下文里找不到/不够长时,从表里链式查
   top-1 后继补足。

## 用法

```bash
pip install -r requirements.txt   # 实测版本:torch 2.14.1+cu130 / transformers 5.21.0 / accelerate 1.15.0

# 1. 建表(语料可以是任何 jsonl/纯文本,递归抽取所有字符串字段;越大命中越高)
python build_ngram_table.py \
  --corpus 语料1.jsonl 语料2.jsonl \
  --out strata_ngram.bin --max-tokens 120000000
# 表建议放到 NVMe/SSD 上,--table 指过去即可

# 2. 聊天(默认纯文本多轮;模型路径用 --model 指向本地目录或 HF id)
python strata_chat.py --interactive
python strata_chat.py --image photo.jpg --text "详细描述这张照片"

# 3. 实验模式
python strata_chat.py --check     # greedy 自证:baseline 与 strata 输出逐 token 一致
python strata_chat.py --bench     # baseline / pld / strata 三模式测速
python strata_chat.py --table-io ram   # 对应 Strata 的 --ple-io ram(整表进内存)

# draft 收益最大的场景:重复上下文
python strata_chat.py --bench --greedy --image "" \
  --text "请一字不差地重复下面方括号里的内容:[...一段话...]"
```

模型:默认 `--model ../LFM3-VL-2B`(本地目录);也可传 HF hub id。
表格式与统计口径见 `build_ngram_table.py` 顶部注释。

## 实测(RTX 5090D 32G,transformers 5.21.0)

58M token 中文语料建表:150 万条 bigram,3.1 MB 文件,41.8% 行覆盖。

| 场景 | baseline | 仅 prompt-lookup | strata(prompt lookup + 表) |
|---|---|---|---|
| 描述照片(开放式生成,greedy) | 55.9 tok/s | 52.9 | 52.1(接受率 2%) |
| 逐字复述 190 字段落(greedy) | 59.1 tok/s | **308.0** | **303.8(接受率 64%,5.1×)** |

一致性:greedy 下 strata 与 baseline 输出逐 token 相同 ✅

### 表的 IO 分层(每次生成结束自动打印 `[IO]` 行)

| 复述任务,greedy,--max-new-tokens 128 | 表热(页缓存自然驻留) | 表冷(`--cold-table`,生成前 fadvise DONTNEED) |
|---|---|---|
| strata 生成速度 | 318.8 tok/s | 318.3 tok/s(**不变**) |
| 查表速度 | 694,595 rows/s | 60,410 rows/s(慢 11.5×,真回盘) |
| 本次生成真实磁盘读(read_bytes) | 0 B | **1.2 MiB** |
| 页缓存驻留(生成结束时) | 792 KiB / 3.0 MiB(26%) | 732 KiB / 3.0 MiB(24%) |

三个可以亲眼看到的分层事实:

- 表 3 MB,**只有被查询过的页驻留**(26%——offsets 数组局部 + 少量数据行),
  其余部分从不进内存;
- `--cold-table` 丢掉全部缓存后,被查询的页**真的从盘上读了 1.2 MiB**
  (major fault 计数受内核 fault-around/readahead 记账影响,以 read_bytes 为准);
- 即便全冷,每轮 draft 只查几十行,回盘开销亚毫秒,**生成速度纹丝不动** ——
  表读 I/O 从来不是瓶颈,这正是"把最冷的大表留在 SSD"成立的理由;
  Strata 那张 28.8 GB 的表同样只按行读,只是行数多得多。

## 结论(与 Strata 文档互相印证)

1. **n-gram draft 只在"重复上下文"的任务上赚**(复述、代码编辑、引用改写):
   上下文里已有答案的形状,抄就完了。开放式创作(描述照片)接受率接近 0,
   draft 反而拖慢 —— 每轮要多验证几个 token。
2. **repetition_penalty 是 draft 的天敌**:它压低"上下文里出现过的 token"的
   概率,而 draft 恰恰全是从上下文抄来的,导致草稿在验证阶段全部被拒。
   想看 draft 收益,用 greedy 或去掉惩罚。
3. **表放 SSD 的代价被 OS 页缓存抹平**:mmap 按需调页,热行驻内存、冷行留盘,
   内存吃紧时内核自动逐出。`--cold-table` 可以强制观察全回盘的最冷情形。

## 文件

- `build_ngram_table.py` —— 语料 → bigram top-K 表(mmap 二进制,行 = 前驱 token)
- `strata_chat.py` —— chat 主程序(mmap 表 + 自定义 CandidateGenerator + assisted generation)
