const BASE = '/api'

// Held only in memory (a module-level variable), never localStorage/sessionStorage,
// so a stolen XSS payload can't read a persisted token — it only lives as long as
// the tab does, and a refresh requires logging in again.
let authToken = null
let onUnauthorized = null

export function setAuthToken(token) {
  authToken = token
}

export function clearAuthToken() {
  authToken = null
}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler
}

async function request(path, options = {}) {
  const authHeader = authToken ? { Authorization: `Bearer ${authToken}` } : {}
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: options.body instanceof FormData
      ? { ...authHeader, ...options.headers }
      : { 'Content-Type': 'application/json', ...authHeader, ...options.headers },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail || detail
    } catch { /* ignore */ }
    if (res.status === 401 && onUnauthorized) onUnauthorized()
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  // Auth
  register: (email, password) =>
    request('/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) }),
  login: (email, password) =>
    request('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  me: () => request('/auth/me'),

  // Documents
  listDocuments: () => request('/documents'),
  getDocument: (id) => request(`/documents/${id}`),
  getPages: (id) => request(`/documents/${id}/pages`),
  deleteDocument: (id) => request(`/documents/${id}`, { method: 'DELETE' }),
  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },

  // Ask / chat
  askDocument: (id, question, conversationId) =>
    request(`/documents/${id}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question, conversation_id: conversationId || null }),
    }),
  getMessages: (docId, convId) => request(`/documents/${docId}/conversations/${convId}/messages`),

  // Ask / chat — streaming (Server-Sent Events over a chunked POST response).
  // onEvent receives parsed {type: 'conversation'|'delta'|'done', ...} objects.
  askDocumentStream: async (id, question, conversationId, onEvent, signal) => {
    const authHeader = authToken ? { Authorization: `Bearer ${authToken}` } : {}
    const res = await fetch(`${BASE}/documents/${id}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeader },
      body: JSON.stringify({ question, conversation_id: conversationId || null }),
      signal,
    })
    if (res.status === 401 && onUnauthorized) onUnauthorized()
    if (!res.ok || !res.body) {
      throw new Error(`Streaming request failed (${res.status})`)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const payload = trimmed.slice(5).trim()
        if (!payload) continue
        onEvent(JSON.parse(payload))
      }
    }
  },

  // Search
  searchDocument: (id, query, method = 'hybrid', top_k = 8) =>
    request(`/documents/${id}/search`, {
      method: 'POST',
      body: JSON.stringify({ query, method, top_k }),
    }),

  // Analysis
  getEntities: (id) => request(`/documents/${id}/entities`),
  getKeywords: (id) => request(`/documents/${id}/keywords`),
  getPos: (id) => request(`/documents/${id}/pos`),
  getStatistics: (id) => request(`/documents/${id}/statistics`),
  getToc: (id) => request(`/documents/${id}/toc`),

  // Summary
  summarize: (id, length = 'standard') =>
    request(`/documents/${id}/summarize`, {
      method: 'POST',
      body: JSON.stringify({ length }),
    }),

  // Comparison
  compareDocuments: (idA, idB) =>
    request('/compare', {
      method: 'POST',
      body: JSON.stringify({ document_id_a: idA, document_id_b: idB }),
    }),

  // Knowledge graph
  getGraph: (id) => request(`/documents/${id}/graph`),

  // Evaluation
  runEvaluation: (documentId, evalSet, compareRerank = false) =>
    request('/evaluation/run', {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId, eval_set: evalSet, compare_rerank: compareRerank }),
    }),

  // Workspaces
  listWorkspaces: () => request('/workspaces'),
  getWorkspace: (id) => request(`/workspaces/${id}`),
  createWorkspace: (name) =>
    request('/workspaces', { method: 'POST', body: JSON.stringify({ name }) }),
  deleteWorkspace: (id) => request(`/workspaces/${id}`, { method: 'DELETE' }),
  addWorkspaceDocuments: (id, documentIds) =>
    request(`/workspaces/${id}/documents`, {
      method: 'POST',
      body: JSON.stringify({ document_ids: documentIds }),
    }),
  removeWorkspaceDocument: (id, documentId) =>
    request(`/workspaces/${id}/documents/${documentId}`, { method: 'DELETE' }),
  askWorkspace: (id, question) =>
    request(`/workspaces/${id}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),

  // Settings
  getSettings: () => request('/settings'),
  setGeminiKey: (apiKey) =>
    request('/settings/gemini-key', { method: 'POST', body: JSON.stringify({ api_key: apiKey }) }),
  testConnection: () => request('/settings/test-connection', { method: 'POST' }),
}
