"""
Token usage optimization utilities for premium workflows.
Includes context compression, prompt optimization, and quality-preserving balancing.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OptimizedContext:
    content: str
    original_tokens: int
    optimized_tokens: int
    quality_score: float
    compression_ratio: float
    metadata: Dict[str, Any]


@dataclass
class CompressedPrompt:
    content: str
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float


class TokenCounter:
    def count_tokens(self, text: str) -> int:
        # Rough proxy; replace with tokenizer
        return max(1, len(text.split()))


class QualityPreserver:
    def score_quality(self, text: str) -> float:
        # Heuristic quality scoring
        if not text:
            return 0.0
        completeness = min(len(text) / 500.0, 1.0)
        structure = 0.1 if any(k in text.lower() for k in ["because", "therefore", "however", "for example"]) else 0.0
        return max(0.0, min(1.0, 0.6 * completeness + structure + 0.2))


class ContextCompressor:
    def compress(self, text: str, target_tokens: int) -> str:
        words = text.split()
        if len(words) <= target_tokens:
            return text
        return " ".join(words[:target_tokens])


class PromptOptimizer:
    def optimize(self, prompt: str, target_tokens: int) -> str:
        # Simple cleanup: remove redundancy and extra whitespace
        cleaned = " ".join(prompt.split())
        words = cleaned.split()
        if len(words) <= target_tokens:
            return cleaned
        return " ".join(words[:target_tokens])


class TokenOptimizer:
    def __init__(self):
        self.context_compressor = ContextCompressor()
        self.prompt_optimizer = PromptOptimizer()
        self.token_counter = TokenCounter()
        self.quality_preserver = QualityPreserver()

    async def optimize_context_window(self, context: str, max_tokens: int) -> OptimizedContext:
        original_tokens = self.token_counter.count_tokens(context)
        compressed_text = self.context_compressor.compress(context, max_tokens)
        optimized_tokens = self.token_counter.count_tokens(compressed_text)
        quality = self.quality_preserver.score_quality(compressed_text)
        ratio = optimized_tokens / max(original_tokens, 1)
        return OptimizedContext(
            content=compressed_text,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            quality_score=quality,
            compression_ratio=ratio,
            metadata={"method": "truncate_words"},
        )

    async def compress_prompt(self, prompt: str, target_tokens: int) -> CompressedPrompt:
        original_tokens = self.token_counter.count_tokens(prompt)
        optimized = self.prompt_optimizer.optimize(prompt, target_tokens)
        optimized_tokens = self.token_counter.count_tokens(optimized)
        ratio = optimized_tokens / max(original_tokens, 1)
        return CompressedPrompt(
            content=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            compression_ratio=ratio,
        )

    async def balance_quality_cost(self, content: str, quality_threshold: float = 0.75) -> OptimizedContext:
        # Try multiple targets and pick best quality-cost balance
        original_tokens = self.token_counter.count_tokens(content)
        best: OptimizedContext | None = None
        for ratio in (0.5, 0.6, 0.7, 0.8, 0.9):
            target = max(1, int(original_tokens * ratio))
            candidate = await self.optimize_context_window(content, target)
            if candidate.quality_score >= quality_threshold:
                best = candidate
                break
            if best is None or candidate.quality_score > best.quality_score:
                best = candidate
        return best  # type: ignore


