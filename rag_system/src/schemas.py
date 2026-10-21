from __future__ import annotations
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel


"""
Схемы для данных RAG системы.
- BlockType: Типы блоков
- Block: Блок текста
- ChunkMetadata: Метаданные чанка
- Chunk: Чанок
- EmbeddedChunk: Векторизованный чанок
- RetrievedChunk: Найденный чанок

Блок текста:
- type: Тип блока
- content: Содержимое блока
- language: Язык кода
- listing_number: Номер листинга

Метаданные чанка:
- book: Название книги
- author: Автор книги
- chapter: Номер главы
- chapter_title: Заголовок главы
- section: Номер раздела
- section_title: Заголовок раздела
- block_type: Тип блока
- language: Язык кода
- listing_number: Номер листинга
- page: Номер страницы

Чанк:
- chunk_id: ID чанк
- text: Текст чанк
- metadata: Метаданные чанк

Векторизованный чанк:
- chunk_id: ID чанк
- text: Текст чанка
- embedding: Вектор чанк
- metadata: Метаданные чанк

Найденный чанк с результатом поиска:
- chunk_id: ID чанк
- text: Текст чанк
- score: Score чанк
- metadata: Метаданные чанк    
"""

# Хуй
class BlockType(str, Enum):
    PART       = "part"
    CHAPTER    = "chapter"
    SECTION    = "section"
    CODE       = "code"
    TABLE      = "table"
    DEFINITION = "definition"
    EXAMPLE    = "example"
    NOTE       = "note"
    IMAGE      = "image"
    TEXT       = "text"


class Block(BaseModel):
    type: BlockType
    content: str
    language: Optional[str] = None
    listing_number: Optional[str] = None


class ChunkMetadata(BaseModel):
    book: str
    author: Optional[str] = None
    part_title: str = ""
    chapter_title: str = ""
    section_title: str = ""
    block_type: BlockType
    language: Optional[str] = None
    listing_number: Optional[str] = None
    page: Optional[int] = None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class EmbeddedChunk(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
