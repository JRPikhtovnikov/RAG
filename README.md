# RAG System

Система для интеллектуального поиска и вопросно-ответного взаимодействия с документами (PDF). Загружает книги/документы, разбивает на смысловые фрагменты, векторизует и отвечает на вопросы с указанием источников.

## Стек

| Слой | Технология |
|------|-----------|
| Backend | FastAPI + Python 3.11 |
| Frontend | React + Vite + TypeScript |
| База данных | PostgreSQL 16 |
| Поисковый движок | Elasticsearch 9 |
| Эмбеддинги | Yandex Embeddings API |
| LLM | YandexGPT |
| Реранкер | Sentence-Transformers (BAAI/bge-reranker-v2-m3) |
| Деплой | Docker Compose |

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                     Frontend                        │
│  React SPA: чаты, загрузка документов, поиск        │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│                  FastAPI Backend                     │
│                                                     │
│  /query       — задать вопрос (RAG-пайплайн)        │
│  /upload      — загрузить PDF                       │
│  /documents   — список документов                   │
│  /chats       — история чатов                       │
│  /auth        — регистрация / вход                  │
│  /code-review — ревью кода                          │
└──────────┬────────────────────┬─────────────────────┘
           │                    │
┌──────────▼──────┐   ┌─────────▼──────────┐
│   PostgreSQL    │   │   Elasticsearch    │
│                 │   │                    │
│  users          │   │  семантический     │
│  chats          │   │  поиск (kNN)       │
│  chat_messages  │   │                    │
│  query_history  │   │  BM25 full-text    │
│  user_settings  │   │  (Russian analyzer)│
└─────────────────┘   └────────────────────┘
```

---

## RAG-пайплайн

### Индексация документа

```
PDF
 └→ PyMuPDF → Markdown
               └→ Chunker (разбивка на фрагменты с метаданными)
                   └→ Yandex Embeddings API → векторы [256 чисел]
                                               └→ Elasticsearch
```

### Ответ на вопрос

```
Вопрос
 ├→ Yandex Embeddings API → вектор запроса
 │                           └→ kNN поиск (top-40)  ─┐
 └→ BM25 full-text поиск (top-40)                    ─┤
                                                      ▼
                              Гибридное слияние (0.7 семантика + 0.3 BM25)
                                                      │
                                    (опц.) Реранкер CrossEncoder
                                                      │
                              top-5 фрагментов → контекст → YandexGPT
                                                                │
                                                        Ответ + источники
```

---

## Контекст диалога

В чате система учитывает последние сообщения при поиске и формировании ответа.
Например, после «Что такое Unit of Work?» можно спросить «А чем он отличается от Repository?»:
LLM сначала уточняет поисковый запрос с учётом истории, затем запускается обычный RAG-пайплайн.
Исходный вопрос сохраняется в чате и передаётся модели вместе с историей и найденными фрагментами.

История загружается с сервера только для выбранного чата после проверки владельца.
По умолчанию используются последние 12 сообщений (обычно 6 пар вопрос–ответ), суммарно до 12 000 символов.
При превышении лимита старые реплики исключаются, а слишком длинная последняя пара сокращается.
Автоматическое резюме более ранней переписки пока не реализовано.

Первый вопрос не требует дополнительного вызова LLM. Для последующих вопросов добавляется
вызов переписывания; при ошибке или таймауте поиск продолжится с исходным вопросом.
История помогает понять вопрос, а подтверждением фактов остаются текущие фрагменты документов.
В новом чате история начинается заново; `/query/` остаётся независимым одиночным запросом.

Статус планируемых и реализованных функций: [PLANNED_FEATURES.md](PLANNED_FEATURES.md).

---

## Компоненты

### Backend (`rag_system/src/`)

| Файл / Папка | Назначение |
|---|---|
| `main.py` | Точка входа FastAPI, инициализация сервисов |
| `config.py` | Настройки из `.env` через Pydantic Settings |
| `vectorstore.py` | Elasticsearch: индексация, kNN + BM25, гибридное слияние |
| `schemas.py` | Pydantic-схемы: `Chunk`, `EmbeddedChunk`, `RetrievedChunk` |
| `models.py` | SQLAlchemy-модели БД |
| `db.py` | Инициализация PostgreSQL |
| `auth.py` | JWT-аутентификация |
| `services/embeding_service.py` | Генерация эмбеддингов через Yandex Embeddings API |
| `services/llm_service.py` | Обращение к YandexGPT для генерации ответа |
| `services/query_service.py` | Оркестратор RAG-пайплайна |
| `services/chunker_service.py` | Разбивка Markdown на фрагменты с иерархией |
| `services/reranker_service.py` | CrossEncoder реранкинг (Sentence-Transformers) |
| `services/document_conversion_service.py` | PDF → Markdown (PyMuPDF) |
| `services/data_loading_service.py` | Оркестратор индексации |
| `services/code_review_service.py` | Ревью кода через LLM |
| `routes/` | FastAPI-роутеры для всех endpoint'ов |
| `promts/` | Шаблоны системных промптов |

### Frontend (`frontend/src/`)

| Страница / Компонент | Назначение |
|---|---|
| `AuthPage` | Вход и регистрация |
| `ChatPage` | Чат с RAG-системой, просмотр источников |
| `DashboardPage` | Главная панель |
| `DocumentsPage` | Управление загруженными документами |
| `SettingsPage` | Настройки пользователя |
| `ChatPanel` + `ChatComposer` | UI чата |
| `SourceCard` | Карточка источника с фрагментом текста |

---

## Переменные окружения

Создай файл `rag_system/.env`:

```env
# Yandex Cloud API
OPENAI_API_KEY=<yandex-iam-token-or-api-key>
OPENAI_BASE_URL=https://llm.api.cloud.yandex.net/v1/

# Модели (замени <folder-id> на ID каталога в Yandex Cloud)
LLM_MODEL=gpt://<folder-id>/yandexgpt/latest
EMBEDDING_MODEL_DOC=emb://<folder-id>/text-search-doc/latest
EMBEDDING_MODEL_QUERY=emb://<folder-id>/text-search-query/latest
EMBEDDING_DIM=256

# Elasticsearch
ES_URL=http://localhost:9200
ES_INDEX=rag_chunks

# Параметры поиска
TOP_K=5                   # финальных источников в ответе
RETRIEVAL_K=40            # после гибридного слияния
DENSE_K=40                # топ kNN-результатов
SPARSE_K=40               # топ BM25-результатов
NUM_CANDIDATES=300        # kNN-кандидаты
HYBRID_DENSE_WEIGHT=0.7   # вес семантики (0..1)

# Контекст диалога (необязательно, указаны значения по умолчанию)
CHAT_HISTORY_MESSAGES=12          # число последних сообщений; 0 отключает историю
CHAT_HISTORY_MAX_CHARS=12000      # максимальное число символов истории
CHAT_REWRITE_TIMEOUT_SECONDS=10   # таймаут уточнения поискового запроса

# Параметры чанкинга
MAX_TOKENS_CHUNK=800
CHUNK_OVERLAP=50
EMBEDDING_CONCURRENCY=1

# Реранкер
ENABLE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_BATCH_SIZE=16

# Аутентификация
JWT_SECRET=change-me-to-random-secret
JWT_EXPIRE_HOURS=720

# База данных (переопределяется Docker Compose)
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/rag_app
```

> `OPENAI_API_KEY` и `OPENAI_BASE_URL` используются для совместимости с OpenAI SDK — Yandex Foundation Models поддерживает этот протокол.

---

## Запуск

### Docker Compose (рекомендуется)

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd RAG

# 2. Создать .env файл
cp rag_system/.env.example rag_system/.env
# Заполнить OPENAI_API_KEY и <folder-id> в названиях моделей

# 3. Запустить все сервисы
docker compose up --build

# Приложение: http://localhost:5173
# API:        http://localhost:8000
# ES:         http://localhost:9200
```

> При первом запуске Docker скачает образы (~2 ГБ) и соберёт контейнеры.  
> Если включён реранкер (`ENABLE_RERANKER=true`), при первом старте загрузится модель (~1 ГБ) — кешируется в Docker volume `hf_cache`.

### Локальная разработка

**Backend:**
```bash
cd RAG

# Зависимости
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Запустить только инфраструктуру
docker compose up postgres elasticsearch -d

# Запустить backend
cd rag_system
uvicorn src.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

---

## API

| Метод | Endpoint | Описание |
|---|---|---|
| `POST` | `/auth/register` | Регистрация |
| `POST` | `/auth/login` | Вход, возвращает JWT |
| `POST` | `/query/` | Задать вопрос (RAG) |
| `POST` | `/upload/` | Загрузить PDF |
| `GET` | `/documents/` | Список документов |
| `GET` | `/chats/` | История чатов |
| `POST` | `/chats/` | Создать чат |
| `GET` | `/chats/{id}/messages` | Сообщения чата |
| `POST` | `/chats/{id}/ask` | Вопрос с учётом истории текущего чата |
| `POST` | `/code-review/` | Ревью кода |
| `GET` | `/profile/` | Профиль пользователя |

Интерактивная документация: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Тесты контекста диалога

Запуск из корня проекта в окружении с установленными backend-зависимостями:

```bash
PYTHONPATH=rag_system python -m unittest discover -s rag_system/tests -v
```

Проверяются ограничение и порядок истории, уточнение поискового запроса, обработка сбоев,
передача ролей в LLM, HTTP-обработчик чата и проверка доступа. Тесты используют подменённые
сервисы и сессии: PostgreSQL, Elasticsearch и обращения к Yandex API не требуются.

---

## Зависимости

```
fastapi==0.121.0
uvicorn==0.38.0
elasticsearch==9.1.3
openai==2.7.0
SQLAlchemy==2.0.36
asyncpg==0.30.0
pydantic>=2,<3
pydantic-settings>=2,<3
pymupdf4llm==0.0.20           # PDF → Markdown
sentence-transformers==5.1.2  # реранкер
bcrypt==4.2.0
PyJWT==2.10.1
python-multipart==0.0.20
aiohttp==3.9.5
```
