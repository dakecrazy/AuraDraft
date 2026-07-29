# DEPRECATED: 被 tools/ops.py 替代，保留仅为 M3/M4 测试
"""LLM client abstraction with a domain-aware Mock for testing.

M3: The LLM is the "deep reasoning" layer — it reads documents, compares
them to cluster knowledge cards, and makes classification decisions.
The Mock simulates this with keyword-based domain detection so tests
prove the architecture without API calls.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract interface for LLM interactions in the knowledge base."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a text response from *prompt* (with optional *system*)."""
        ...

    def generate_json(self, prompt: str, schema_hint: str = "") -> dict[str, Any]:
        """Generate a JSON-structured response.

        Default implementation: generate text, strip markdown fences, parse JSON.
        Subclasses may override for structured-output APIs.
        """
        text = self.generate(prompt, system=schema_hint)
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


# ── Domain-aware Mock ──────────────────────────────────────────────

# Keywords → domain label (used by MockLLMClient)
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "深度学习": [
        "深度学习", "注意力", "attention", "transformer", "梯度",
        "神经网络", "训练", "模型", "学习率", "dropout", "embedding",
        "损失函数", "反向传播", "激活函数", "卷积", "循环神经网络",
    ],
    "法律合同": [
        "法律合同", "合同", "甲方", "乙方", "违约", "赔偿", "诉讼",
        "条款", "采购", "付款", "争议", "盖章", "签订",
    ],
    "量子计算": [
        "量子计算", "量子", "qubit", "量子比特", "纠缠", "量子门",
        "Shor", "Grover", "量子纠错", "叠加态",
    ],
    "烹饪食谱": [
        "烹饪食谱", "食谱", "翻炒", "焯水", "收汁", "炖煮", "食材",
        "五花肉", "冰糖", "生抽", "老抽",
    ],
}


class MockLLMClient(LLMClient):
    """Domain-aware mock that classifies documents by keyword overlap.

    This is the most important test artifact — it proves the M3
    architecture works without real API calls.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt: str, system: str = "") -> str:
        self.call_count += 1

        # Use unique section markers to route (NOT fragile substring combos)
        if "【候选知识领域】" in prompt:
            return self._mock_classify(prompt)
        if "当前知识档案" in prompt and "新文档带来的知识增量" in prompt:
            return self._mock_update_card(prompt)
        if "【核心命题】" in prompt or "深度阅读" in prompt:
            return self._mock_deep_read(prompt)
        if "知识领域标签" in prompt or "简短的知识领域" in prompt:
            return self._mock_generate_label(prompt)
        if "初始知识档案" in prompt:
            return self._mock_generate_card(prompt)
        if "参考资料" in prompt and "问题" in prompt:
            return self._mock_synthesize(prompt)
        if "摘要" in prompt or "概括" in prompt:
            return self._mock_summarize(prompt)

        return "Mock response"

    def generate_json(self, prompt: str, schema_hint: str = "") -> dict[str, Any]:
        text = self.generate(prompt, system=schema_hint)
        # If the mock returned a dict as string, parse it
        if isinstance(text, str):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: return as text field
        return {"text": text}

    # ── mock internals ────────────────────────────────────────────

    def _detect_domain(self, text: str) -> str:
        """Score each domain by keyword overlap, return the best match."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(text_lower.count(kw.lower()) for kw in keywords)
            if score > 0:
                scores[domain] = score
        if not scores:
            return "未分类"
        return max(scores, key=scores.get)

    def _mock_deep_read(self, prompt: str) -> str:
        """Simulate deep reading — extract keywords from the prompt."""
        # Find the document text in the prompt (after the last "文档：")
        match = re.search(r"文档[：:]\s*(.+?)(?:\n\n|\Z)", prompt, re.DOTALL)
        doc_text = match.group(1)[:500] if match else ""
        domain = self._detect_domain(doc_text)

        return json.dumps(
            {
                "核心命题": f"这篇文档主要讨论{domain}领域的相关内容",
                "知识要素": f"涉及{domain}的核心概念和方法论",
                "增量": "提供了该领域的最新进展和关键技术细节",
                "隐含推断": "这些技术可能与其他领域产生交叉应用",
                "碰撞点": f"与{domain}的其他子方向有潜在关联",
            },
            ensure_ascii=False,
        )

    def _mock_classify(self, prompt: str) -> str:
        """Simulate classification — find the best matching cluster."""
        # Extract document text
        doc_match = re.search(r"【新文档】\s*(.+?)(?:\n\n【候选|$)", prompt, re.DOTALL)
        doc_text = doc_match.group(1)[:500] if doc_match else ""
        domain = self._detect_domain(doc_text)

        # Extract cluster labels from the prompt
        cluster_labels = re.findall(r"=== (\S+) ===", prompt)
        cluster_info = {}
        for label in cluster_labels:
            # Find the card text for this cluster
            card_match = re.search(
                rf"=== {re.escape(label)} ===\s*(.+?)(?:\n\n===|\Z)", prompt, re.DOTALL
            )
            card_text = card_match.group(1)[:200] if card_match else ""
            cluster_info[label] = card_text

        # Find the best matching cluster
        best_cluster = None
        best_score = -1
        for cid, card in cluster_info.items():
            card_domain = self._detect_domain(card)
            if card_domain == domain:
                score = 10  # exact domain match
            else:
                score = 1  # weak match
            if score > best_score:
                best_score = score
                best_cluster = cid

        if best_cluster and best_score >= 5:
            return json.dumps(
                {
                    "primary_cluster": best_cluster,
                    "reasoning": f"文档主题与{domain}领域高度相关",
                    "knowledge_delta": f"补充了{domain}领域的新技术细节",
                    "cross_links": [],
                    "new_cluster_suggestion": None,
                    "card_update": f"在{domain}领域下新增相关技术条目",
                },
                ensure_ascii=False,
            )
        else:
            # Create new cluster
            return json.dumps(
                {
                    "primary_cluster": "__new__",
                    "reasoning": f"文档主题{domain}不属于任何现有领域",
                    "knowledge_delta": f"创建新的{domain}领域知识档案",
                    "cross_links": [],
                    "new_cluster_suggestion": domain,
                    "card_update": f"初始化{domain}领域知识档案",
                },
                ensure_ascii=False,
            )

    def _mock_update_card(self, prompt: str) -> str:
        """Simulate card update — return a slightly modified version.

        Uses a more specific regex to avoid truncating at the first \\n\\n
        inside the existing card content.
        """
        # Find existing card content between "当前知识档案" and "新文档带来的知识增量"
        card_match = re.search(
            r"当前知识档案（已积累 \d+ 篇文档）[：:]\s*(.+?)(?:\n\n新文档带来的知识增量|\Z)",
            prompt, re.DOTALL
        )
        existing = card_match.group(1).strip()[:200] if card_match else ""
        return f"{existing}\n\n## 更新记录\n- 新增内容: 补充了相关领域的最新进展"

    def _mock_synthesize(self, prompt: str) -> str:
        """Simulate answer synthesis from reference chunks."""
        doc_match = re.search(
            r"参考资料[：:]\s*(.+?)(?:\n\n问题|\Z)", prompt, re.DOTALL
        )
        context = doc_match.group(1)[:500] if doc_match else ""
        domain = self._detect_domain(context)
        sources = re.findall(r"\[来源: (.+?)\]", prompt)
        source_str = ", ".join(sources[:3]) if sources else "知识库"
        return (
            f"基于{domain}相关资料："
            f"根据知识库中的文档，{domain}领域涉及多个关键技术概念。"
            f"来源：{source_str}"
        )

    def _mock_generate_label(self, prompt: str) -> str:
        """Generate a 2-6 character label from the document text."""
        doc_match = re.search(r"文档[：:]\s*(.+?)(?:\n\n|\Z)", prompt, re.DOTALL)
        doc_text = doc_match.group(1)[:500] if doc_match else ""
        domain = self._detect_domain(doc_text)
        return domain if domain and domain != "未分类" else "未分类"

    def _mock_generate_card(self, prompt: str) -> str:
        """Generate an initial knowledge_card that includes domain keywords."""
        label_match = re.search(r"「(.+?)」", prompt)
        label = label_match.group(1) if label_match else "未知领域"
        doc_match = re.search(r"文档[：:]\s*(.+?)(?:\n\n输出|\Z)", prompt, re.DOTALL)
        doc_text = doc_match.group(1)[:500] if doc_match else ""
        domain = self._detect_domain(doc_text)
        # CRITICAL: card text MUST contain domain keywords so _detect_domain()
        # can match it when classifying subsequent documents
        return (
            f"领域：{domain}\n\n"
            f"核心知识：涉及{domain}的关键概念和方法论\n\n"
            f"知识演进：当前的技术状态和最新进展\n\n"
            f"未解决的问题：基于文档内容的开放性问题\n\n"
            f"代表性文档：初始文档作为起点"
        )

    def _mock_summarize(self, prompt: str) -> str:
        """Simulate summarization."""
        doc_match = re.search(r"文档[：:]\s*(.+?)(?:\n\n|\Z)", prompt, re.DOTALL)
        doc_text = doc_match.group(1)[:200] if doc_match else ""
        domain = self._detect_domain(doc_text)
        return f"这是一篇关于{domain}的技术文档，涵盖了核心概念和最新进展。"