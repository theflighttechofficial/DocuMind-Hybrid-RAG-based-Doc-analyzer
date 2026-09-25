import { useEffect, useRef, useState } from 'react'
import { Layers, Plus, Trash2, Send, X } from 'lucide-react'
import { api } from '../services/api.js'
import EvidenceCard, { sortByRelevance } from '../components/EvidenceCard.jsx'
import EmptyState from '../components/EmptyState.jsx'

export default function Workspaces() {
  const [workspaces, setWorkspaces] = useState([])
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeId, setActiveId] = useState(null)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  const refresh = () =>
    Promise.all([api.listWorkspaces(), api.listDocuments()]).then(([ws, docs]) => {
      setWorkspaces(ws)
      setDocuments(docs.filter((d) => d.processing_status === 'ready'))
      setLoading(false)
    })

  useEffect(() => { refresh() }, [])

  const active = workspaces.find((w) => w.id === activeId) || null

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    try {
      const ws = await api.createWorkspace(newName.trim())
      setNewName('')
      await refresh()
      setActiveId(ws.id)
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Delete this workspace? Documents themselves are not deleted.')) return
    await api.deleteWorkspace(id)
    if (activeId === id) setActiveId(null)
    refresh()
  }

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (active) {
    return (
      <WorkspaceDetail
        workspace={active}
        allDocuments={documents}
        onBack={() => setActiveId(null)}
        onChanged={refresh}
      />
    )
  }

  return (
    <div className="animate-fade-in">
      <h1 className="font-display text-2xl text-paper mb-1">Workspaces</h1>
      <p className="text-muted text-sm mb-6">
        Group documents together and ask questions across all of them at once.
      </p>

      <form onSubmit={handleCreate} className="mb-8 flex gap-2 max-w-md">
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New workspace name…"
          className="flex-1 bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-4 py-2.5 text-sm text-paper placeholder:text-muted focus:border-signal outline-none"
        />
        <button
          type="submit"
          disabled={creating || !newName.trim()}
          className="px-4 rounded-xl bg-signal text-ink disabled:opacity-40 disabled:cursor-not-allowed hover:bg-paper transition-colors flex items-center gap-1.5"
        >
          <Plus size={16} /> Create
        </button>
      </form>

      {workspaces.length === 0 ? (
        <EmptyState
          icon={Layers}
          title="No workspaces yet."
          description="Create a workspace above, then add documents to it for cross-document Ask."
        />
      ) : (
        <div className="grid md:grid-cols-2 gap-3">
          {workspaces.map((ws) => (
            <div
              key={ws.id}
              className="relative group rounded-xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5 cursor-pointer hover:border-signalDim transition-colors"
              onClick={() => setActiveId(ws.id)}
            >
              <div className="font-display text-base text-paper mb-1">{ws.name}</div>
              <div className="text-xs text-muted font-mono">
                {ws.documents.length} document{ws.documents.length === 1 ? '' : 's'}
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); handleDelete(ws.id) }}
                className="absolute top-4 right-4 p-1.5 rounded text-muted opacity-0 group-hover:opacity-100 hover:text-rust hover:bg-ink/40 transition-all"
                title="Delete workspace"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function WorkspaceDetail({ workspace, allDocuments, onBack, onChanged }) {
  const memberIds = new Set(workspace.documents.map((d) => d.id))
  const available = allDocuments.filter((d) => !memberIds.has(d.id))
  const [picking, setPicking] = useState(false)
  const [selectedToAdd, setSelectedToAdd] = useState([])
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const toggleSelect = (id) => {
    setSelectedToAdd((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])
  }

  const handleAddSelected = async () => {
    if (selectedToAdd.length === 0) return
    await api.addWorkspaceDocuments(workspace.id, selectedToAdd)
    setSelectedToAdd([])
    setPicking(false)
    onChanged()
  }

  const handleRemove = async (docId) => {
    await api.removeWorkspaceDocument(workspace.id, docId)
    onChanged()
  }

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    const q = question.trim()
    setQuestion('')
    setMessages((prev) => [...prev, { role: 'user', content: q }])
    setAsking(true)
    try {
      const res = await api.askWorkspace(workspace.id, q)
      setMessages((prev) => [...prev, { role: 'assistant', content: res.answer, evidence: res.evidence, grounded: res.grounded }])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: err.message || 'Something went wrong.', evidence: [], grounded: false }])
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="animate-fade-in flex flex-col h-[calc(100vh-8rem)] md:h-[calc(100vh-6rem)]">
      <div className="flex items-center justify-between gap-4 mb-4">
        <div>
          <button onClick={onBack} className="text-xs font-mono text-muted hover:text-paper mb-1">← all workspaces</button>
          <h1 className="font-display text-xl text-paper">{workspace.name}</h1>
        </div>
        <button
          onClick={() => setPicking((p) => !p)}
          className="text-xs font-mono text-signal hover:text-paper border border-line/15 rounded-xl px-3 py-1.5"
        >
          {picking ? 'close' : '+ add documents'}
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {workspace.documents.length === 0 && (
          <span className="text-xs text-muted">No documents in this workspace yet.</span>
        )}
        {workspace.documents.map((d) => (
          <span key={d.id} className="flex items-center gap-1.5 text-xs font-mono bg-raised text-paper px-2.5 py-1 rounded-xl">
            {d.name}
            <button onClick={() => handleRemove(d.id)} className="text-muted hover:text-rust">
              <X size={12} />
            </button>
          </span>
        ))}
      </div>

      {picking && (
        <div className="mb-4 rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-4">
          {available.length === 0 ? (
            <div className="text-xs text-muted">No other ready documents available to add.</div>
          ) : (
            <>
              <div className="grid sm:grid-cols-2 gap-2 mb-3">
                {available.map((d) => (
                  <label key={d.id} className="flex items-center gap-2 text-sm text-paper/85 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedToAdd.includes(d.id)}
                      onChange={() => toggleSelect(d.id)}
                      className="accent-signal"
                    />
                    {d.name}
                  </label>
                ))}
              </div>
              <button
                onClick={handleAddSelected}
                disabled={selectedToAdd.length === 0}
                className="text-xs font-mono px-3 py-1.5 rounded-xl bg-signal text-ink disabled:opacity-40 disabled:cursor-not-allowed hover:bg-paper transition-colors"
              >
                Add {selectedToAdd.length || ''} selected
              </button>
            </>
          )}
        </div>
      )}

      {workspace.documents.length === 0 ? (
        <EmptyState title="Add documents to this workspace to start asking cross-document questions." />
      ) : (
        <>
          <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-6 pr-1">
            {messages.length === 0 && (
              <div className="text-muted text-sm py-8 text-center">
                Ask a question across all documents in this workspace.
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={m.role === 'user' ? 'flex justify-end' : ''}>
                {m.role === 'user' ? (
                  <div className="max-w-[80%] rounded-2xl bg-raised px-4 py-2.5 text-sm text-paper">
                    {m.content}
                  </div>
                ) : (
                  <div className="max-w-[85%] space-y-3">
                    <div className={`text-sm leading-relaxed ${m.grounded ? 'text-paper' : 'text-amber'}`}>
                      {m.content}
                    </div>
                    {m.evidence?.length > 0 && (
                      <div className="space-y-3 pt-1">
                        <div className="text-[11px] text-muted font-mono uppercase tracking-wide">Evidence</div>
                        {sortByRelevance(m.evidence).map((ev, j) => (
                          <EvidenceCard key={ev.citation ?? j} {...ev} documentName={ev.document_name} />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {asking && (
              <div className="flex gap-1 text-muted text-sm">
                <span className="animate-pulse-dot">●</span>
                <span className="animate-pulse-dot" style={{ animationDelay: '0.2s' }}>●</span>
                <span className="animate-pulse-dot" style={{ animationDelay: '0.4s' }}>●</span>
              </div>
            )}
          </div>

          <form onSubmit={handleAsk} className="mt-4 flex gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask anything across this workspace's documents…"
              className="flex-1 bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-4 py-3 text-sm text-paper placeholder:text-muted focus:border-signal outline-none"
            />
            <button
              type="submit"
              disabled={asking || !question.trim()}
              className="px-4 rounded-xl bg-signal text-ink disabled:opacity-40 disabled:cursor-not-allowed hover:bg-paper transition-colors"
            >
              <Send size={16} />
            </button>
          </form>
        </>
      )}
    </div>
  )
}
