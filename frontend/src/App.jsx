import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_URL = ''
const MODELS = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it']

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [model, setModel] = useState(MODELS[0])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [sidebarTab, setSidebarTab] = useState('chats')
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    fetchSessions()
  }, [])

  useEffect(() => {
    if (sessionId) fetchDocuments()
  }, [sessionId])

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_URL}/api/sessions`)
      const data = await res.json()
      setSessions(data)
      if (data.length > 0 && !sessionId) {
        loadSession(data[0].id)
      }
    } catch (err) {
      console.error('Failed to fetch sessions:', err)
    }
  }

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/api/documents?session_id=${sessionId}`)
      const data = await res.json()
      setDocuments(data)
    } catch (err) {
      console.error('Failed to fetch documents:', err)
    }
  }

  const loadSession = async (id) => {
    try {
      const res = await fetch(`${API_URL}/api/sessions/${id}/messages`)
      const data = await res.json()
      setMessages(data.map(m => ({ role: m.role, content: m.content, sources: m.sources || [] })))
      setSessionId(id)
    } catch (err) {
      console.error('Failed to load session:', err)
    }
  }

  const newChat = () => {
    setMessages([])
    setSessionId(null)
    setDocuments([])
  }

  const deleteSession = async (e, id) => {
    e.stopPropagation()
    if (!confirm('Удалить чат?')) return
    try {
      await fetch(`${API_URL}/api/sessions/${id}`, { method: 'DELETE' })
      fetchSessions()
      if (id === sessionId) {
        setMessages([])
        setSessionId(null)
        setDocuments([])
      }
    } catch (err) {
      console.error('Failed to delete session:', err)
    }
  }

  const uploadFile = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const sid = sessionId || Date.now().toString()
    if (!sessionId) setSessionId(sid)

    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_URL}/api/upload?session_id=${sid}`, {
        method: 'POST',
        body: formData
      })
      if (res.ok) {
        fetchDocuments()
      }
    } catch (err) {
      console.error('Upload failed:', err)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const deleteDocument = async (docId) => {
    if (!confirm('Удалить документ?')) return
    try {
      await fetch(`${API_URL}/api/documents/${docId}`, { method: 'DELETE' })
      fetchDocuments()
    } catch (err) {
      console.error('Failed to delete document:', err)
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages,
          model,
          session_id: sessionId
        })
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let assistantMessage = ''
      let sources = []

      setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [] }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.content) {
                assistantMessage += data.content
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantMessage,
                    sources
                  }
                  return updated
                })
              }
              if (data.sources) {
                sources = data.sources
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: assistantMessage,
                    sources
                  }
                  return updated
                })
              }
              if (data.error) {
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    content: 'Ошибка: ' + data.error,
                    sources: []
                  }
                  return updated
                })
              }
              if (data.session_id) {
                setSessionId(data.session_id)
                fetchSessions()
              }
            } catch (err) {}
          }
        }
      }
    } catch (err) {
      console.error('Chat error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={newChat}>
            + New Chat
          </button>
        </div>
        <div className="sidebar-tabs">
          <button
            className={`sidebar-tab ${sidebarTab === 'chats' ? 'active' : ''}`}
            onClick={() => setSidebarTab('chats')}
          >
            Чаты
          </button>
          <button
            className={`sidebar-tab ${sidebarTab === 'docs' ? 'active' : ''}`}
            onClick={() => setSidebarTab('docs')}
          >
            Документы
          </button>
        </div>
        {sidebarTab === 'chats' ? (
          <div className="sessions-list">
            {sessions.map(session => (
              <div
                key={session.id}
                className={`session-item ${sessionId === session.id ? 'active' : ''}`}
                onClick={() => loadSession(session.id)}
              >
                <span className="session-title">{session.title}</span>
                <button
                  className="session-delete"
                  onClick={(e) => deleteSession(e, session.id)}
                  title="Удалить"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="docs-list">
            {!sessionId && <p className="docs-hint">Откройте чат для управления документами</p>}
            {sessionId && (
              <>
                <label className="upload-btn">
                  <input
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={uploadFile}
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                  />
                  {uploading ? 'Загрузка...' : '+ Загрузить файл'}
                </label>
                {documents.map(doc => (
                  <div key={doc.id} className="doc-item">
                    <span className="doc-name">{doc.filename}</span>
                    <button
                      className="doc-delete"
                      onClick={() => deleteDocument(doc.id)}
                      title="Удалить"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </aside>
      <main className="main">
        <div className="model-selector">
          <select value={model} onChange={e => setModel(e.target.value)}>
            {MODELS.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>AI Chat + RAG</h2>
              <p>Загрузите документы и задавайте вопросы по ним</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? 'U' : 'AI'}
              </div>
              <div className="message-body">
                <div className="message-content">
                  {msg.content}
                  {loading && i === messages.length - 1 && msg.role === 'assistant' && !msg.content && (
                    <span className="typing-indicator">...</span>
                  )}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <div className="sources-label">Источники:</div>
                    {msg.sources.map((src, j) => (
                      <div key={j} className="source-item">
                        <span className="source-filename">{src.filename}</span>
                        <span className="source-excerpt">{src.excerpt}...</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <form className="input-form" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App