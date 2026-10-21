"""Run from the repository root: PYTHONPATH=rag_system .venv/bin/python -m unittest discover -s rag_system/tests -v."""

import asyncio
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

# Unit and HTTP tests use fake services, never production credentials or databases.
with patch.dict(os.environ, {
    "OPENAI_API_KEY": "test", "EMBEDDING_API_KEY": "test",
    "OPENAI_BASE_URL": "http://llm.invalid/v1", "LLM_MODEL": "test-model",
    "EMBEDDING_MODEL_DOC": "test-doc", "EMBEDDING_MODEL_QUERY": "test-query",
    "ES_URL": "http://es.invalid:9200", "ES_INDEX": "test", "EMBEDDING_DIM": "2",
    "MAX_TOKENS_CHUNK": "800", "CHUNK_OVERLAP": "50", "TOP_K": "5",
    "RETRIEVAL_K": "40", "DENSE_K": "40", "SPARSE_K": "40",
    "NUM_CANDIDATES": "300", "HYBRID_DENSE_WEIGHT": "0.7",
    "EMBEDDING_CONCURRENCY": "1", "ENABLE_RERANKER": "false",
    "RERANKER_MODEL": "test", "RERANKER_BATCH_SIZE": "16", "JWT_SECRET": "test",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/test",
    "CHAT_HISTORY_MESSAGES": "12", "CHAT_HISTORY_MAX_CHARS": "12000",
    "CHAT_REWRITE_TIMEOUT_SECONDS": "10",
}):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.auth import get_current_user
    from src.config import settings
    from src.db import get_session
    from src.routes.chats import _load_recent_history, router
    from src.schemas import BlockType, ChunkMetadata, ConversationMessage, RetrievedChunk
    from src.services.code_review_service import CodeReviewService
    from src.services.conversation_history import limit_history
    from src.services.llm_service import LLMService
    from src.services.query_service import QueryService


def message(role, content):
    return ConversationMessage(role=role, content=content)


class HistoryTests(unittest.TestCase):
    def test_recent_complete_turns_are_kept_in_order(self):
        history = [
            message("user", "old question"), message("assistant", "old answer"),
            message("user", "recent question"), message("assistant", "recent answer"),
        ]
        self.assertEqual(limit_history(history, 2, 1000), history[-2:])
        # A window cut in the middle of a turn must not start with an assistant.
        self.assertEqual(limit_history(history, 3, 1000), history[-2:])

    def test_character_budget_keeps_the_latest_turn_and_does_not_modify_history(self):
        history = [message("user", "Q" * 100), message("assistant", "A" * 1000)]
        result = limit_history(history, 12, 100)
        self.assertEqual([m.role for m in result], ["user", "assistant"])
        self.assertLessEqual(sum(len(m.content) for m in result), 100)
        self.assertEqual(len(history[1].content), 1000)

    def test_character_budget_drops_older_turns(self):
        history = [
            message("user", "q" * 30), message("assistant", "a" * 30),
            message("user", "last"), message("assistant", "answer"),
        ]
        self.assertEqual(limit_history(history, 12, 20), history[-2:])

    def test_empty_or_disabled_history(self):
        self.assertEqual(limit_history([], 12, 100), [])
        self.assertEqual(limit_history([message("user", "question")], 0, 100), [])
        self.assertEqual(limit_history([message("user", "   ")], 12, 100), [])
        self.assertEqual(limit_history([message("assistant", "orphan")], 12, 100), [])


class QueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.history = [
            message("user", "Что такое Unit of Work?"),
            message("assistant", "Unit of Work управляет транзакцией."),
        ]
        self.chunk = RetrievedChunk(
            chunk_id="1", text="Unit of Work и Repository решают разные задачи.", score=0.9,
            metadata=ChunkMetadata(book="Architecture", block_type=BlockType.TEXT),
        )
        self.store = SimpleNamespace(search=AsyncMock(return_value=[self.chunk]))
        self.embeddings = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1, 0.2]))
        self.llm = SimpleNamespace(answer=AsyncMock())
        self.reranker = SimpleNamespace(rerank=AsyncMock(return_value=[self.chunk]))
        self.service = QueryService(self.store, self.embeddings, self.llm, self.reranker)

    async def test_follow_up_is_rewritten_for_all_retrieval_stages(self):
        question = "А чем он отличается от Repository?"
        rewritten = "Чем Unit of Work отличается от Repository?"
        self.llm.answer.side_effect = [rewritten, "Они решают разные задачи."]

        answer, chunks = await self.service.ask(question, history=self.history)

        self.embeddings.embed_query.assert_awaited_once_with(rewritten)
        self.assertEqual(self.store.search.await_args.kwargs["query_text"], rewritten)
        self.reranker.rerank.assert_awaited_once_with(rewritten, [self.chunk])
        final_call = self.llm.answer.await_args_list[-1].kwargs
        self.assertEqual(final_call["question"], question)
        self.assertEqual(final_call["history"], self.history)
        self.assertIn(self.chunk.text, final_call["prompt"])
        self.assertEqual(answer, "Они решают разные задачи.")
        self.assertEqual(chunks, [self.chunk])

    async def test_first_question_does_not_make_a_rewrite_call(self):
        self.llm.answer.return_value = "Ответ по книге."
        await self.service.ask("Что такое Repository?")
        self.llm.answer.assert_awaited_once()
        self.embeddings.embed_query.assert_awaited_once_with("Что такое Repository?")
        self.assertEqual(self.llm.answer.await_args.kwargs["history"], [])

    async def test_standalone_topic_change_can_keep_the_original_query(self):
        question = "Что такое TDD?"
        self.llm.answer.side_effect = [question, "Ответ про TDD."]
        await self.service.ask(question, self.history)
        self.embeddings.embed_query.assert_awaited_once_with(question)

    async def test_rewrite_errors_empty_or_oversized_results_fall_back(self):
        for result in [RuntimeError("unavailable"), "  ", "x" * 2001]:
            with self.subTest(result=type(result).__name__):
                self.embeddings.embed_query.reset_mock()
                self.llm.answer.side_effect = [result, "Ответ по фрагментам."]
                answer, _ = await self.service.ask("А пример?", self.history)
                self.embeddings.embed_query.assert_awaited_once_with("А пример?")
                self.assertEqual(answer, "Ответ по фрагментам.")
                self.assertEqual(self.llm.answer.await_args.kwargs["history"], self.history)

    async def test_rewrite_timeout_does_not_fail_the_answer(self):
        calls = 0

        async def delayed_answer(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(1)
            return "Ответ по фрагментам."

        self.llm.answer.side_effect = delayed_answer
        with patch.object(settings, "chat_rewrite_timeout_seconds", 0.01):
            answer, _ = await self.service.ask("А пример?", self.history)
        self.assertEqual(answer, "Ответ по фрагментам.")
        self.embeddings.embed_query.assert_awaited_once_with("А пример?")

    async def test_bounded_history_is_used_for_rewrite_and_answer(self):
        history = self.history + [message("user", "new"), message("assistant", "new answer")]
        self.llm.answer.side_effect = ["Уточнённый запрос", "Ответ."]
        with patch.object(settings, "chat_history_messages", 2):
            await self.service.ask("Уточни", history)
        for call in self.llm.answer.await_args_list:
            self.assertEqual(call.kwargs["history"], history[-2:])

    async def test_history_is_not_shared_between_calls(self):
        self.llm.answer.side_effect = ["Запрос с темой", "Первый ответ", "Второй ответ"]
        await self.service.ask("Уточни", self.history)
        await self.service.ask("Новый чат")
        self.assertEqual(self.llm.answer.await_count, 3)
        self.assertEqual(self.llm.answer.await_args.kwargs["history"], [])

    async def test_no_context_answer_still_removes_sources(self):
        self.llm.answer.side_effect = ["Переписанный запрос", "В контексте нет ответа."]
        _, chunks = await self.service.ask("Уточни", self.history)
        self.assertEqual(chunks, [])


class LLMTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ"))],
        ))
        self.llm = LLMService(SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )))

    async def test_history_is_sent_as_ordered_messages_with_original_roles(self):
        history = [message("user", "First question"), message("assistant", "First answer")]
        await self.llm.answer("System rules", "Follow up", history)
        self.assertEqual(self.create.await_args.kwargs["messages"], [
            {"role": "system", "content": "System rules"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Вопрос пользователя: Follow up"},
        ])

    async def test_code_review_works_without_history(self):
        self.assertEqual(await CodeReviewService(self.llm).review("print(1)", "test.py"), "Ответ")
        self.assertEqual(len(self.create.await_args.kwargs["messages"]), 2)


class ChatRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_query_is_scoped_and_ordered_with_a_limit(self):
        history = [message("user", "question"), message("assistant", "answer")]
        result = Mock()
        result.all.return_value = list(reversed(history))
        session = SimpleNamespace(execute=AsyncMock(return_value=result))
        loaded = await _load_recent_history(session, 17, SimpleNamespace(id=42))
        self.assertEqual(loaded, history)
        query = session.execute.await_args.args[0]
        compiled = query.compile(compile_kwargs={"literal_binds": True})
        sql = str(compiled)
        self.assertIn("chat_messages.chat_id = 17", sql)
        self.assertIn("chats.user_id = 42", sql)
        self.assertIn("chat_messages.created_at DESC, chat_messages.id DESC", sql)
        self.assertIn("LIMIT 12", sql)

    async def test_disabled_history_does_not_load_messages(self):
        session = SimpleNamespace(execute=AsyncMock())
        with patch.object(settings, "chat_history_messages", 0):
            self.assertEqual(await _load_recent_history(session, 17, SimpleNamespace(id=42)), [])
        session.execute.assert_not_awaited()


class ChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.session = Mock()
        self.session.execute = AsyncMock()
        self.session.commit = AsyncMock()
        self.session.refresh = AsyncMock(side_effect=self.refresh)
        self.query_service = SimpleNamespace(ask=AsyncMock(return_value=("Ответ", [])))
        self.app.state.query_service = self.query_service
        self.user = SimpleNamespace(id=42)

        async def override_session():
            yield self.session

        self.app.dependency_overrides[get_session] = override_session
        self.app.dependency_overrides[get_current_user] = lambda: self.user

    async def refresh(self, instance):
        if instance.id is None:
            instance.id = 100 if instance.role == "user" else 101
            instance.created_at = datetime.now(timezone.utc)

    def test_ask_loads_saved_history_and_saves_the_original_question(self):
        now = datetime.now(timezone.utc)
        chat = SimpleNamespace(id=17, title="Existing", created_at=now, updated_at=now)
        history = [message("user", "Что такое UoW?"), message("assistant", "Unit of Work.")]
        chat_result, history_result, settings_result, count_result = [Mock() for _ in range(4)]
        chat_result.scalar_one_or_none.return_value = chat
        history_result.all.return_value = list(reversed(history))
        settings_result.scalar_one_or_none.return_value = SimpleNamespace(save_history=True)
        count_result.scalar_one.return_value = 4
        self.session.execute.side_effect = [chat_result, history_result, settings_result, count_result]

        with TestClient(self.app) as client:
            response = client.post("/chats/17/ask", json={"question": "А пример?"})

        self.assertEqual(response.status_code, 200, response.text)
        self.query_service.ask.assert_awaited_once_with("А пример?", history=history)
        self.assertEqual(response.json()["user_message"]["content"], "А пример?")
        self.session.commit.assert_awaited_once()

    def test_inaccessible_chat_is_rejected_before_loading_history(self):
        result = Mock()
        result.scalar_one_or_none.return_value = None
        self.session.execute.return_value = result
        with TestClient(self.app) as client:
            response = client.post("/chats/999/ask", json={"question": "Уточни"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.session.execute.await_count, 1)
        self.query_service.ask.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
