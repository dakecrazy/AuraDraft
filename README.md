# kb-agent

**Token-based knowledge base with MoE-inspired routing.**

kb-agent 是一个知识库基础设施层（零 LLM 调用），暴露 8 个原子工具给 Hermes agent 编排。Hermes 负责"理解"环节（读文档、做分类决策、写 knowledge_card、发现跨域联系），kb-agent 负责"存储和检索"环节（tokenize、索引、聚类、搜索、归档）。

## 架构概览

```
Hermes（推理层 — agent）
  │
  ├── kb_ingest(file)        → doc_id          ← 索引文档
  ├── kb_prefilter(doc_id)   → [候选簇]         ← 统计预筛
  ├── kb_get_cards(cids)     → {card_text}      ← 读知识档案
  │
  ├── [Hermes 决策：归入现有簇 or 创建新簇]
  │
  ├── kb_assign(doc_id, cid)                    ← 归入现有簇
  │   └── kb_archive(file, label)               ← 物理归档
  │
  ├── kb_create(label, card, doc_id) → cid      ← 创建新簇
  │   └── kb_archive(file, label)               ← 物理归档
  │
  ├── kb_update_card(cid, card_text)            ← 更新知识档案
  ├── kb_search(query)        → [搜索结果]       ← BM25 检索
  └── kb_archive(file, label) → archived_path   ← 物理归档
```

### 为什么是 Token 索引？

传统知识库用词级倒排索引（需要 jieba 分词）或向量索引（需要 embedding 模型）。本方案直接用 LLM 的 tokenizer（o200k_base）作为索引单元：

- **天然子词粒度**：不需要额外分词器
- **与 LLM 对齐**：索引和推理在同一个 token 空间
- **整数 ID 查找**：比字符串匹配快一个数量级
- **Token 预算可控**：天然知道每个 chunk 多少 token

### 为什么是 MoE 路由？

当知识库有 10000 个簇时，不可能把所有 knowledge_card 塞进 LLM 上下文。MoE 分层路由解决这个问题：

- **L0（统计预筛）**：token 频率签名 → 余弦相似度 → 选出 Top-K 候选簇（免费，毫秒级）
- **L1（LLM 深度分类）**：只把 Top-K 的 knowledge_card 塞进 prompt → LLM 做理解、推理、归类
- **效果**：无限容量 × 有限计算，和 MoE 模型用 2/64 激活率跑万亿参数是同一个原理

### Tokenizer 兼容策略

索引层用 o200k_base（GPT-4o 的 tokenizer，200K 词表，中文覆盖好，完全开源）。推理层把文本原样传给任何 LLM API。两层之间永远用"文本"做桥梁。换模型时不需要重建索引。

## 8 个原子工具

所有工具通过 CLI 调用：`python -m kb_agent.tools.cli <cmd> <args>`

| 工具 | 用途 | 是否调 LLM |
|------|------|:---:|
| `kb_ingest` | 索引文档（tokenize + chunk + 倒排索引） | ❌ |
| `kb_prefilter` | 统计预筛（token 签名 → 余弦相似度 → Top-K 候选簇） | ❌ |
| `kb_get_cards` | 读知识档案 | ❌ |
| `kb_assign` | 归入现有簇（更新质心 + 知识档案） | ❌ |
| `kb_create` | 创建新簇（label + card + 第一篇文档） | ❌ |
| `kb_update_card` | 更新知识档案 | ❌ |
| `kb_search` | BM25 混合检索（exact × 0.6 + phrase × 0.4） | ❌ |
| `kb_archive` | 物理归档到 `knowledge_base/{label}/` | ❌ |

### 1. kb_ingest — 索引文档

```bash
python -m kb_agent.tools.cli ingest <file> [doc_id] [category]
```

将文档 tokenize、切 chunk、建三层索引（倒排 + bigram + chunk）。**不分类、不归档。** 返回 `doc_id`。

### 2. kb_prefilter — 统计预筛

```bash
python -m kb_agent.tools.cli prefilter <doc_id>
```

从 `doc_signatures` 表读取文档的 token 签名，和所有簇质心算余弦相似度，返回 Top-K 候选簇。当簇数 ≤ top_k 时返回全部。

```json
[{"cluster_id": "...", "label": "深度学习", "similarity": 0.1379, "doc_count": 1}]
```

### 3. kb_get_cards — 读知识档案

```bash
python -m kb_agent.tools.cli get-cards <cid> [cid ...]
```

### 4. kb_assign — 归入现有簇

```bash
cat card.txt | python -m kb_agent.tools.cli assign <doc_id> <cluster_id>
```

card_text 从 stdin 读取（支持多行中文）。自动更新 token doc-frequency。

### 5. kb_create — 创建新簇

```bash
cat card.txt | python -m kb_agent.tools.cli create <label> <doc_id>
```

### 6. kb_update_card — 更新知识档案

```bash
echo "新的知识档案内容" | python -m kb_agent.tools.cli update-card <cluster_id>
```

### 7. kb_search — BM25 检索

```bash
python -m kb_agent.tools.cli search <query> [top_k] [mode]
```

模式：`exact`（倒排）、`phrase`（bigram）、`hybrid`（默认，0.6+0.4）。

### 8. kb_archive — 物理归档

```bash
python -m kb_agent.tools.cli archive <file> <label> [doc_id]
```

复制文件到 `knowledge_base/{label}/`。自动更新索引中的 `file_path`。

## 编排工作流

### 标准流程（每篇文档）

```bash
# 1. 索引
python -m kb_agent.tools.cli ingest paper.txt doc_001

# 2. 预筛
python -m kb_agent.tools.cli prefilter doc_001
# → []（第一篇文档）

# 3. 创建簇（Hermes 生成 label + card）
cat <<'CARD' | python -m kb_agent.tools.cli create "领域名" doc_001
领域：领域名
核心知识：...
CARD

# 4. 归档
python -m kb_agent.tools.cli archive paper.txt "领域名" doc_001
```

### 查询流程

```bash
python -m kb_agent.tools.cli search "注意力机制的计算复杂度"
# → [dl_001: 15.81, dl_002: 1.44, legal_001: 0.24]
```

### 批量摄入（100+ 篇文档）

在单个 `terminal()` 中运行 Python 脚本避免每次加载 tiktoken：

```python
from kb_agent.tools import init_kb, kb_ingest, kb_prefilter, kb_assign, kb_create, kb_archive
from kb_agent.document.loader import iter_documents

session = init_kb("kb_index.db")
session.connect()

for f in iter_documents("./docs/"):
    r = kb_ingest(session, str(f))
    doc_id = r["doc_id"]
    candidates = kb_prefilter(session, doc_id)
    if candidates:
        cid = candidates[0]["cluster_id"]
        kb_assign(session, doc_id, cid)
    else:
        kb_create(session, "新领域", "初始知识档案", doc_id)
    kb_archive(session, str(f), "新领域")

session.close()
```

## 安装

```bash
git clone https://github.com/dakecrazy/kb-agent.git
cd kb-agent
pip install -e .
```

依赖：`tiktoken`、`numpy`（自动安装）。

### Hermes Skill 注册

```bash
ln -s /path/to/kb-agent ~/.hermes/skills/data-science/kb-agent
```

然后在 Hermes 对话中 `/skill kb-agent` 加载。

## 数据存储

| 数据 | 位置 | 说明 |
|------|------|------|
| 索引 + 簇 | `kb_index.db` | SQLite，WAL 模式 |
| 物理文件 | `knowledge_base/{label}/` | 归档后的文档副本 |
| Token 缓存 | `.cache/tiktoken/` | BPE 文件缓存 |

## 真实编排演示

以下是一次完整的 Hermes 编排过程（3 篇文档，2 个领域）：

```
# 1. 索引 dl_001（注意力机制综述）
$ python -m kb_agent.tools.cli ingest tests/fixtures/sample_docs/deep_learning_attention.txt dl_001
→ doc_id=dl_001, tokens=464

# 2. 预筛 → 空（第一篇文档）
$ python -m kb_agent.tools.cli prefilter dl_001
→ []

# 3. Hermes 读文档 → 创建「深度学习」簇
$ cat <<CARD | python -m kb_agent.tools.cli create "深度学习" dl_001
领域：深度学习 - 注意力机制
【核心知识】自注意力 Q/K/V、多头注意力、复杂度 O(n²)
【知识演进】2017 Transformer → Flash Attention
CARD
→ cluster_id=76bf7f85

# 4. 索引 legal_001（买卖合同）
$ python -m kb_agent.tools.cli ingest tests/fixtures/sample_docs/legal_contract.txt legal_001

# 5. 预筛 → 深度学习簇（相似度 0.0105，明显不匹配）
$ python -m kb_agent.tools.cli prefilter legal_001
→ [{cluster_id: 76bf7f85, similarity: 0.0105}]

# 6. Hermes 判断 → 创建「法律合同」簇
$ cat <<CARD | python -m kb_agent.tools.cli create "法律合同" legal_001
领域：法律合同 - 买卖合同
CARD
→ cluster_id=73ba330e

# 7. 索引 dl_002（训练优化技术）
$ python -m kb_agent.tools.cli ingest tests/fixtures/sample_docs/deep_learning_training.txt dl_002

# 8. 预筛 → 深度学习 0.1379，法律合同 0.0119
$ python -m kb_agent.tools.cli prefilter dl_002
→ [{深度学习: 0.1379}, {法律合同: 0.0119}]

# 9. Hermes 判断 → 归入深度学习簇，更新知识档案
$ cat <<CARD | python -m kb_agent.tools.cli assign dl_002 76bf7f85
领域：深度学习
【核心知识】注意力机制 + 训练优化（学习率调度、混合精度、AdamW）
CARD
→ doc_count=2, card_updated=true

# 10. 搜索验证
$ python -m kb_agent.tools.cli search "注意力机制的计算复杂度"
→ dl_001: 15.81, dl_002: 1.44, legal_001: 0.24
```

## 关键设计决策

### 为什么用 CLI 桥接而不是 Python import？

Hermes skill 的范式是工具调用，每个工具是一个独立的 `terminal()` 命令。CLI 调用让 Hermes 不需要管理 Python session 生命周期。对于批量场景，提供 Python 脚本模板在单个 `terminal()` 中执行。

### 为什么 signature 存 DB 而不是返回给调用方？

Signature 是 128 个 token_id→weight 对，约 2KB。如果每篇文档都返回给 Hermes，100 篇文档就是 200KB 上下文浪费。存 DB 后调用方只传 `doc_id`（12 字节），signature 在 `kb_prefilter` 内部读取。

### 为什么 knowledge_card 存 SQLite？

跨会话持久化。Hermes 的上下文是会话级的，重启后丢失。SQLite 中的 knowledge_card 在重启后仍然可用，且和簇绑定。

## 项目结构

```
kb-agent/
├── src/kb_agent/
│   ├── tokenizer/canonical.py   # 规范 Tokenizer（o200k_base）
│   ├── document/loader.py       # 文档加载器
│   ├── index/engine.py          # Token 索引引擎（BM25）
│   ├── cluster/
│   │   ├── model.py             # KnowledgeCluster 数据模型
│   │   └── manager.py           # TokenClusterEngine（统计聚类）
│   ├── storage/
│   │   ├── db.py                # SQLite 连接管理 + schema
│   │   └── cluster_store.py     # 簇持久化
│   ├── tools/
│   │   ├── session.py           # KnowledgeBaseSession
│   │   ├── ops.py               # 8 个原子工具
│   │   └── cli.py               # CLI 入口（推荐使用）
│   ├── pipeline/                # [DEPRECATED] 旧管线代码
│   ├── llm/                     # [DEPRECATED] 旧 LLM 抽象
│   └── cli/                     # [DEPRECATED] 旧 CLI
├── tests/
│   ├── test_index_engine.py     # M1: 索引 + BM25
│   ├── test_cluster_engine.py   # M2: 统计聚类
│   ├── test_m3_pipeline.py      # M3: 管线
│   ├── test_m4_query.py         # M4: 查询
│   └── test_tools.py            # 8 工具编排测试
├── SKILL.md                     # Hermes skill 注册文件
├── pyproject.toml
└── README.md
```

## 里程碑

| 里程碑 | 功能 | 状态 |
|--------|------|------|
| M1 | Token 索引引擎（BM25 搜索 + 持久化） | ✅ |
| M2 | 统计聚类（降噪 + TF 签名 + 余弦相似度） | ✅ |
| M3 | LLM 深度分类（knowledge_card + MoE 路由） | ✅ |
| M4 | 查询管线（BM25 + LLM 合成 + 簇导航） | ✅ |
| M5 | CLI + 物理归档 + 批量摄入 | ✅ |
| Phase 1 | 8 原子工具 + Hermes 编排 | ✅ |

## Pitfalls

- **CLI 每次调用加载 tiktoken ~2 秒** — 批量摄入用 Python 脚本模板
- **`kb_ingest` 后必须 `kb_assign` 或 `kb_create`** — 否则文档只有索引没有归属
- **`kb_archive` 复制文件，不移动** — 源文件保留
- **`kb_prefilter` 在簇数 ≤ top_k 时返回全部** — 不设相似度门槛
- **`kb_update_card` / `kb_assign` / `kb_create` 从 stdin 读取 card_text** — 用 heredoc 或管道
- **SQLite 文件锁** — 不要同时运行多个 CLI 命令操作同一个 DB

## 许可证

MIT