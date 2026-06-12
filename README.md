# AI Chat App

Fullstack AI chat application with FastAPI backend, React frontend, and Groq API.

## Setup

### Backend

```bash
cd backend

# Создать виртуальное окружение
python -m venv venv

# Активировать (Git Bash / Mac / Linux)
source venv/Scripts/activate

# Установить зависимости
pip install -r requirements.txt

# Получи ключ на https://console.groq.com
# Создай файл .env с содержимым:
# GROQ_API_KEY=gsk_...

# Запуск
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at http://localhost:5173

## Features

- Chat with Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B, or Gemma 2 9B (via Groq)
- RAG: загрузка документов (PDF, TXT, MD)
- Автоматическое чанкинг и эмбеддинг через ChromaDB
- Поиск релевантных фрагментов (top-3) для каждого вопроса
- Источники отображаются под ответом AI
- Каждый чат имеет свою область документов
- Streaming responses (tokens appear as they arrive)
- Conversation history saved to SQLite
- Sidebar with session history
- Model selector dropdown
- Auto-scroll to latest message