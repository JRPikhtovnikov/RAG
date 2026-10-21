from abc import ABC, abstractmethod
from typing import Optional, Sequence

from src.config import settings
from src.schemas import ConversationMessage


class LLMServiceInterface(ABC):

    @abstractmethod
    async def answer(
        self, prompt: str, question: str,
        history: Optional[Sequence[ConversationMessage]] = None,
    ) -> str:
        pass


class LLMService(LLMServiceInterface):
    def __init__(self, client):
        self._client = client

    async def answer(
        self, prompt: str, question: str,
        history: Optional[Sequence[ConversationMessage]] = None,
    ) -> str:
        messages = [{"role": "system", "content": prompt}]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in (history or [])
        )
        messages.append({"role": "user", "content": f"Вопрос пользователя: {question}"})
        chat = await self._client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=messages,
        )
        return chat.choices[0].message.content or "Я не знаю."
