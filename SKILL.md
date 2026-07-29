---
name: kb-agent
description: "Token-based knowledge base with MoE routing — 8 atomic tools for Hermes orchestration."
version: 0.1.0
author: Hermes Agent + User
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [knowledge-base, token-index, moe-routing, document-classification]
    related_skills: []
---

# kb-agent

**Token-based knowledge base with MoE-inspired routing.**

kb-agent 是一个知识库基础设施层（零 LLM 调用），暴露 8 个原子工具给 Hermes 编排。Hermes 负责"理解"环节（读文档、做分类决策、写 knowledge_card、发现跨域联系），kb-agent 负责"存储和检索"环节（tokenize、索引、聚类、搜索、归档）。

## 架构概览

```
Hermes（推理层 — 你）
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

## 8 个原子工具

所有工具通过 CLI 调用：`python -m kb_agent.tools.cli <cmd> <args>`

### 1. kb_ingest — 索引文档

```bash
python -m kb_agent.tools.cli ingest <file> [doc_id] [category]
```

将文档 tokenize、切 chunk、建三层索引（倒排 + bigram + chunk）。**不分类、不归档。** 返回 `doc_id`。

**关键约束：** 不返回 signature（存在 DB 内部，`kb_prefilter` 自动读取）。

### 2. kb_prefilter — 统计预筛

```bash
python -m kb_agent.tools.cli prefilter <doc_id>
```

从 `doc_signatures` 表读取文档的 token 签名，和所有簇质心算余弦相似度，返回 Top-K 候选簇。当簇数 ≤ top_k 时返回全部。

**返回格式：**
```json
[{"cluster_id": "...", "label": "...", "similarity": 0.42, "doc_count": 5}]
```

### 3. kb_get_cards — 读知识档案

```bash
python -m kb_agent.tools.cli get-cards <cid> [cid ...]
```

返回指定簇的 knowledge_card 文本。

### 4. kb_assign — 归入现有簇

```bash
python -m kb_agent.tools.cli assign <doc_id> <cluster_id> [card_text]
```

将文档加入簇（更新质心 + 成员列表）。可选更新 knowledge_card。**自动更新 token doc-frequency。**

### 5. kb_create — 创建新簇

```bash
python -m kb_agent.tools.cli create <label> <doc_id> [card_text]
```

以文档为第一个成员创建新簇。返回 `cluster_id`。

### 6. kb_update_card — 更新知识档案

```bash
echo "新的知识档案内容" | python -m kb_agent.tools.cli update-card <cluster_id>
```

从 stdin 读取 card_text，更新簇的 knowledge_card。

### 7. kb_search — BM25 检索

```bash
python -m kb_agent.tools.cli search <query> [top_k] [mode]
```

模式：`exact`（倒排）、`phrase`（bigram）、`hybrid`（默认，0.6+0.4）。

### 8. kb_archive — 物理归档

```bash
python -m kb_agent.tools.cli archive <file> <label> [doc_id]
```

复制文件到 `knowledge_base/{label}/`。处理文件名冲突。

## 编排工作流

### 标准流程（每篇文档）

```
1. kb_ingest(file)           → doc_id
2. kb_prefilter(doc_id)      → [候选簇] 或 []
3. if 候选簇:
     kb_get_cards([cids])    → 读知识档案
     [Hermes 判断：归入哪个簇？]
     kb_assign(doc_id, cid)  → 归簇
   else:
     [Hermes 生成 label + card]
     kb_create(label, card, doc_id) → 新簇
4. kb_archive(file, label)   → 物理归档
```

### 查询流程

```
1. kb_search(query)          → [搜索结果]
2. [Hermes 读 chunk 文本 + 生成回答]
```

### 批量摄入（100+ 篇文档）

对于批量场景，在单个 `terminal()` 中运行 Python 脚本避免每次加载 tiktoken：

```python
from kb_agent.tools import init_kb, kb_ingest, kb_prefilter, kb_get_cards, kb_assign, kb_create, kb_archive
from kb_agent.document.loader import iter_documents

session = init_kb("kb_index.db")
session.connect()

for f in iter_documents("./docs/"):
    r = kb_ingest(session, str(f))
    doc_id = r["doc_id"]
    candidates = kb_prefilter(session, doc_id)
    if candidates:
        # Hermes 需要判断归入哪个簇
        # 这里简化：归入相似度最高的
        cid = candidates[0]["cluster_id"]
        kb_assign(session, doc_id, cid)
    else:
        kb_create(session, "新领域", "初始知识档案", doc_id)
    kb_archive(session, str(f), "新领域")

session.close()
```

## 关键约束

1. **Hermes 永远不接触 signature** — signature 在 `kb_ingest` 时存入 `doc_signatures` 表，`kb_prefilter` 自动读取。调用方只传 `doc_id`。
2. **knowledge_card 是 Hermes 写入的文本** — 存 SQLite 跨会话持久化。Hermes 负责生成和更新 card 内容。
3. **token doc-frequency 在 assign/create 时更新** — 不在 ingest 时更新，确保 IDF 不包含"自己"。
4. **物理归档后索引路径自动更新** — `kb_archive` 后索引中的 `file_path` 指向归档位置。

## 安装

```bash
cd ~/kb_agent
pip install -e .
```

依赖：`tiktoken`、`numpy`（自动安装）。

## 数据存储

| 数据 | 位置 | 说明 |
|------|------|------|
| 索引 + 簇 | `kb_index.db` | SQLite，WAL 模式 |
| 物理文件 | `knowledge_base/{label}/` | 归档后的文档副本 |
| Token 缓存 | `.cache/tiktoken/` | BPE 文件缓存 |

## 设计决策

### 为什么用 CLI 桥接而不是 Python import？

Hermes skill 的范式是工具调用，每个工具是一个独立的 `terminal()` 命令。CLI 调用让 Hermes 不需要管理 Python session 生命周期。对于批量场景，提供 Python 脚本模板在单个 `terminal()` 中执行。

### 为什么 signature 存 DB 而不是返回给调用方？

Signature 是 128 个 token_id→weight 对，约 2KB。如果每篇文档都返回给 Hermes，100 篇文档就是 200KB 上下文浪费。存 DB 后调用方只传 `doc_id`（12 字节），signature 在 `kb_prefilter` 内部读取。

### 为什么 knowledge_card 存 SQLite？

跨会话持久化。Hermes 的上下文是会话级的，重启后丢失。SQLite 中的 knowledge_card 在重启后仍然可用，且和簇绑定。

## Pitfalls

- **CLI 每次调用加载 tiktoken ~2 秒** — 批量摄入用 Python 脚本模板
- **`kb_ingest` 后必须 `kb_assign` 或 `kb_create`** — 否则文档只有索引没有归属，不会被搜索到
- **`kb_archive` 复制文件，不移动** — 源文件保留
- **`kb_prefilter` 在簇数 ≤ top_k 时返回全部** — 不设相似度门槛
- **`kb_update_card` 从 stdin 读取** — 用 `echo "..." |` 或 heredoc
- **SQLite 文件锁** — 不要同时运行多个 CLI 命令操作同一个 DB