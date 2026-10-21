import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.auth import get_current_user
from src.config import settings
from src.db import get_session
from src.models import Chat, ChatMessage, QueryHistory, User, UserSettings
from src.schemas import ConversationMessage
from src.routes.schemas import (
    ChatAskResponse,
    ChatCreateRequest,
    ChatMessageResponse,
    ChatMessageSource,
    ChatMessagesResponse,
    ChatSummaryResponse,
    QueryRequest,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])


def _has_source_fragment(chunk) -> bool:
    return bool((chunk.text or "").strip() and (chunk.metadata.book or "").strip())


def _source_payload(chunk) -> dict:
    return {
        "fileName": chunk.metadata.book or "",
        "text": chunk.text,
        "score": float(chunk.score),
        "page": chunk.metadata.page,
        "book": chunk.metadata.book or "",
        "author": chunk.metadata.author,
        "part": chunk.metadata.part_title or "",
        "chapter": chunk.metadata.chapter_title or "",
        "section": chunk.metadata.section_title or "",
        "block_type": str(chunk.metadata.block_type) if chunk.metadata.block_type else None,
    }


def _summary(chat: Chat, message_count: int) -> ChatSummaryResponse:
    return ChatSummaryResponse(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at.isoformat(),
        updated_at=chat.updated_at.isoformat(),
        message_count=message_count,
    )


def _message_response(message: ChatMessage) -> ChatMessageResponse:
    sources: List[ChatMessageSource] = []
    if message.sources_json:
        try:
            raw = json.loads(message.sources_json)
            sources = [ChatMessageSource(**item) for item in raw]
        except (ValueError, TypeError):
            logger.warning("Failed to parse sources for message %s", message.id)
    return ChatMessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at.isoformat(),
        sources=sources,
    )


async def _load_chat(session: AsyncSession, chat_id: int, user: User) -> Chat:
    result = await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id)
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat


async def _load_recent_history(
    session: AsyncSession, chat_id: int, user: User,
) -> List[ConversationMessage]:
    if settings.chat_history_messages == 0:
        return []
    result = await session.execute(
        select(ChatMessage.role, ChatMessage.content)
        .join(Chat, Chat.id == ChatMessage.chat_id)
        .where(
            ChatMessage.chat_id == chat_id,
            Chat.user_id == user.id,
            ChatMessage.role.in_(("user", "assistant")),
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(settings.chat_history_messages)
    )
    return [
        ConversationMessage(role=message.role, content=message.content)
        for message in reversed(result.all())
    ]


@router.get("/", response_model=List[ChatSummaryResponse])
async def list_chats(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    msg_count_sq = (
        select(func.count(ChatMessage.id))
        .where(ChatMessage.chat_id == Chat.id)
        .correlate(Chat)
        .scalar_subquery()
    )
    result = await session.execute(
        select(Chat, msg_count_sq)
        .where(Chat.user_id == current_user.id)
        .order_by(desc(Chat.updated_at))
    )
    return [_summary(chat, count or 0) for chat, count in result.all()]


@router.post("/", response_model=ChatSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: ChatCreateRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    chat = Chat(user_id=current_user.id, title=(payload.title or "Новый чат").strip() or "Новый чат")
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return _summary(chat, 0)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    chat = await _load_chat(session, chat_id, current_user)
    await session.delete(chat)
    await session.commit()


@router.get("/{chat_id}/messages", response_model=ChatMessagesResponse)
async def list_messages(
    chat_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    result = await session.execute(
        select(Chat)
        .options(selectinload(Chat.messages))
        .where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

    messages = [_message_response(m) for m in chat.messages]
    return ChatMessagesResponse(chat=_summary(chat, len(messages)), messages=messages)


@router.post("/{chat_id}/ask", response_model=ChatAskResponse)
async def ask_in_chat(
    chat_id: int,
    payload: QueryRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    chat = await _load_chat(session, chat_id, current_user)

    try:
        history = await _load_recent_history(session, chat.id, current_user)
        answer, chunks = await request.app.state.query_service.ask(payload.question, history=history)
    except Exception:
        logger.exception("Error during chat ask")
        raise HTTPException(status_code=500, detail="Internal error")

    sources_payload = [_source_payload(c) for c in chunks if _has_source_fragment(c)]

    user_msg = ChatMessage(
        chat_id=chat.id,
        role="user",
        content=payload.question,
        sources_json=None,
    )
    assistant_msg = ChatMessage(
        chat_id=chat.id,
        role="assistant",
        content=answer,
        sources_json=json.dumps(sources_payload, ensure_ascii=False) if sources_payload else None,
    )
    session.add_all([user_msg, assistant_msg])

    if chat.title in {"", "Новый чат"}:
        chat.title = payload.question[:60].strip() or "Новый чат"

    settings_result = await session.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = settings_result.scalar_one_or_none()
    if user_settings is None or user_settings.save_history:
        session.add(
            QueryHistory(
                user_id=current_user.id,
                question=payload.question,
                answer=answer,
                source_count=len(sources_payload),
            )
        )

    await session.commit()
    await session.refresh(user_msg)
    await session.refresh(assistant_msg)
    await session.refresh(chat)

    count_result = await session.execute(
        select(func.count(ChatMessage.id)).where(ChatMessage.chat_id == chat.id)
    )
    message_count = int(count_result.scalar_one())

    return ChatAskResponse(
        chat=_summary(chat, message_count),
        user_message=_message_response(user_msg),
        assistant_message=_message_response(assistant_msg),
    )
