from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from src.config import settings
from src.schemas import ConversationMessage, RetrievedChunk
from src.services.conversation_history import limit_history
from src.services.embeding_service import EmbeddingServiceInterface
from src.services.llm_service import LLMServiceInterface
from src.promts.prompt_ask import ask_prompt, rewrite_question_prompt

if TYPE_CHECKING:
    from src.services.reranker_service import RerankerServiceInterface


logger = logging.getLogger(__name__)


NO_CONTEXT_MARKERS = (
    "информация отсутствует",
    "В предоставленных фрагментах книги эта информация отсутствует.",
    "в предоставленных фрагментах книги эта информация отсутствует",
    "в предоставленных фрагментах эта информация отсутствует",
    "в контексте нет",
    "не найдено",
    "не нашёл",
    "не нашел",
)


def _answer_has_no_context(answer: str) -> bool:
    normalized = answer.lower().replace("ё", "е")
    return any(marker.replace("ё", "е") in normalized for marker in NO_CONTEXT_MARKERS)


class QueryService:
    def __init__(
        self,
        store,
        embedding_service: EmbeddingServiceInterface,
        llm_service: LLMServiceInterface,
        reranker_service: Optional[RerankerServiceInterface] = None,
    ):
        self._store = store
        self._embedding_service = embedding_service
        self._llm_service = llm_service
        self._reranker_service = reranker_service

    async def ask(
        self, question: str, history: Optional[Sequence[ConversationMessage]] = None,
    ) -> Tuple[str, List[RetrievedChunk]]:
        total_start = time.perf_counter()
        history = limit_history(
            history or [], settings.chat_history_messages, settings.chat_history_max_chars,
        )
        stage_start = time.perf_counter()
        search_question = await self._rewrite_question(question, history) if history else question
        rewrite_ms = (time.perf_counter() - stage_start) * 1000
        embed_ms = search_ms = rerank_ms = llm_ms = 0.0
        retrieved_count = 0
        reranked_count = 0

        stage_start = time.perf_counter()
        query_vector = await self._embedding_service.embed_query(search_question)
        embed_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        chunks = await self._store.search(
            query_text=search_question,
            query_vector=query_vector,
            k=settings.retrieval_k,
            dense_k=settings.dense_k,
            sparse_k=settings.sparse_k,
            num_candidates=settings.num_candidates,
            dense_weight=settings.hybrid_dense_weight,
        )
        search_ms = (time.perf_counter() - stage_start) * 1000
        retrieved_count = len(chunks)

        if self._reranker_service is not None:
            stage_start = time.perf_counter()
            chunks = await self._reranker_service.rerank(search_question, chunks)
            rerank_ms = (time.perf_counter() - stage_start) * 1000
        reranked_count = len(chunks)

        final_chunks = chunks[:settings.top_k]

        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(final_chunks))

        stage_start = time.perf_counter()
        answer = await self._llm_service.answer(
            prompt=ask_prompt(context),
            question=question,
            history=history,
        )
        llm_ms = (time.perf_counter() - stage_start) * 1000

        if _answer_has_no_context(answer):
            final_chunks = []

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "Query processed: history_messages=%s rewrite_ms=%.2f "
            "embed_ms=%.2f search_ms=%.2f rerank_ms=%.2f "
            "llm_ms=%.2f total_ms=%.2f retrieved=%s reranked=%s final=%s",
            len(history),
            rewrite_ms,
            embed_ms,
            search_ms,
            rerank_ms,
            llm_ms,
            total_ms,
            retrieved_count,
            reranked_count,
            len(final_chunks),
        )
        return answer, final_chunks

    async def _rewrite_question(
        self, question: str, history: Sequence[ConversationMessage],
    ) -> str:
        try:
            rewritten = await asyncio.wait_for(
                self._llm_service.answer(
                    prompt=rewrite_question_prompt(), question=question, history=history,
                ),
                timeout=settings.chat_rewrite_timeout_seconds,
            )
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) <= 2000:
                return rewritten
            logger.warning("Question rewrite was empty or too long; using original question")
        except Exception as exc:
            logger.warning("Question rewrite failed (%s); using original question", type(exc).__name__)
        return question
