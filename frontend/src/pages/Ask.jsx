import { useState, useRef, useEffect } from 'react'
import { Send, RotateCcw } from 'lucide-react'
import { api } from '../services/api.js'
import EvidenceCard, { sortByRelevance } from '../components/EvidenceCard.jsx'
import EmptyState from '../components/EmptyState.jsx'
import { MessageSquareText } from 'lucide-react'
import useDocumentPicker, { DocumentPickerBar } from '../hooks/useDocumentPicker.jsx'

export default function Ask() {
  const { documents, selected, selectDocument, loading, error, refresh } = useDocumentPicker('/ask')
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [conversationId, setConversationId] = useState(null)
  const [asking, setAsking] = useState(false)
  const scrollRef = useRef(null)
  const abortRef = useRef(null)

  // Switching documents (or unmounting) mid-stream must not let a stale
  // in-flight response keep writing into the new document's message list —
  // abort it and stop trusting its callbacks.
  useEffect(() => {
    abortRef.current?.abort()
    setMessages([])
    setConversationId(null)
    setAsking(false)
    return () => abortRef.current?.abort()
  }, [selected?.id])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!question.trim() || !selected || asking) return
    const q = question.trim()
    setQuestion('')
    setMessages((prev) => [...prev, { role: 'user', content: q }, { role: 'assistant', content: '', evidence: [], grounded: true, streaming: true }])
    setAsking(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await api.askDocumentStream(selected.id, q, conversationId, (event) => {
        if (controller.signal.aborted) return
        if (event.type === 'conversation') {
          setConversationId(event.conversation_id)
        } else if (event.type === 'delta') {
          setMessages((prev) => {
            if (prev.length === 0) return prev
            const next = [...prev]
            const last = next[next.length - 1]
            next[next.length - 1] = { ...last, content: last.content + event.text }
            return next
          })
        } else if (event.type === 'done') {
          setMessages((prev) => {
            if (prev.length === 0) return prev
            const next = [...prev]
            next[next.length - 1] = {
              role: 'assistant', content: event.answer, evidence: event.evidence, grounded: event.grounded, streaming: false,
            }
            return next
          })
        }
      }, controller.signal)
    } catch (err) {
      if (controller.signal.aborted) return
      setMessages((prev) => {
        if (prev.length === 0) return prev
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', content: err.message || 'Something went wrong. Please try again.', evidence: [], grounded: false, streaming: false }
        return next
      })
    } finally {
      if (!controller.signal.aborted) setAsking(false)
    }
  }

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (error) {
    return (
      <EmptyState
        icon={MessageSquareText}
        title="Couldn't load your documents."
        description={error}
      >
        <button
          onClick={refresh}
          className="inline-flex items-center gap-1.5 mt-5 text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl"
        >
          <RotateCcw size={14} /> Retry
        </button>
      </EmptyState>
    )
  }

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={MessageSquareText}
        title="No documents ready yet."
        description="Upload and process a document before asking questions about it."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  return (
    <div className="animate-fade-in flex flex-col h-[calc(100vh-8rem)] md:h-[calc(100vh-6rem)]">
      <div className="flex items-center justify-between gap-4 mb-4">
        <h1 className="font-display text-xl text-paper">Ask</h1>
        <DocumentPickerBar documents={documents} selectedId={selected?.id} onSelect={selectDocument} />
      </div>

      {!selected ? (
        <EmptyState title="Select a document to begin." />
      ) : (
        <>
          <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-6 pr-1">
            {messages.length === 0 && (
              <div className="text-muted text-sm py-8 text-center">
                Ask anything about {selected.name}.
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
                    <div className={`text-sm leading-relaxed whitespace-pre-wrap ${m.grounded ? 'text-paper' : 'text-amber'}`}>
                      {m.content}
                      {m.streaming && <span className="inline-block w-1.5 h-3.5 bg-signal/70 ml-0.5 animate-pulse align-text-bottom" />}
                    </div>
                    {m.evidence?.length > 0 && (
                      <div className="space-y-3 pt-1">
                        <div className="text-[11px] text-muted font-mono uppercase tracking-wide">Evidence</div>
                        {sortByRelevance(m.evidence).map((ev, j) => (
                          <EvidenceCard key={ev.citation ?? j} {...ev} />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {asking && messages[messages.length - 1]?.content === '' && (
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
              disabled={asking}
              placeholder="Ask anything about this document…"
              className="flex-1 bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-4 py-3 text-sm text-paper placeholder:text-muted focus:border-signal outline-none disabled:opacity-60"
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
