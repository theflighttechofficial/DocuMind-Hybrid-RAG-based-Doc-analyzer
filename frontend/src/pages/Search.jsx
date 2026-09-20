import { useState } from 'react'
import { SearchIcon } from 'lucide-react'
import { api } from '../services/api.js'
import EvidenceCard from '../components/EvidenceCard.jsx'
import EmptyState from '../components/EmptyState.jsx'
import useDocumentPicker, { DocumentPickerBar } from '../hooks/useDocumentPicker.jsx'

const METHODS = [
  { key: 'semantic', label: 'Semantic' },
  { key: 'bm25', label: 'Keyword' },
  { key: 'hybrid', label: 'Hybrid' },
]

export default function Search() {
  const { documents, selected, selectDocument, loading } = useDocumentPicker('/search')
  const [query, setQuery] = useState('')
  const [method, setMethod] = useState('hybrid')
  const [results, setResults] = useState(null)
  const [searching, setSearching] = useState(false)

  const runSearch = async (e) => {
    e?.preventDefault()
    if (!query.trim() || !selected) return
    setSearching(true)
    try {
      const res = await api.searchDocument(selected.id, query, method)
      setResults(res)
    } finally {
      setSearching(false)
    }
  }

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={SearchIcon}
        title="No documents ready yet."
        description="Upload and process a document before searching it."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h1 className="font-display text-xl text-paper">Semantic Search</h1>
        <DocumentPickerBar documents={documents} selectedId={selected?.id} onSelect={selectDocument} />
      </div>

      {!selected ? (
        <EmptyState title="Select a document to search." />
      ) : (
        <>
          <form onSubmit={runSearch} className="flex gap-2 mb-4">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search naturally — exact wording isn't required…"
              className="flex-1 bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-4 py-3 text-sm text-paper placeholder:text-muted focus:border-signal outline-none"
            />
            <button
              type="submit"
              disabled={searching || !query.trim()}
              className="px-5 rounded-xl bg-signal text-ink text-sm font-mono disabled:opacity-40"
            >
              Search
            </button>
          </form>

          <div className="flex gap-1 mb-6">
            {METHODS.map((m) => (
              <button
                key={m.key}
                onClick={() => setMethod(m.key)}
                className={`text-xs font-mono px-3 py-1.5 rounded-xl transition-colors ${
                  method === m.key ? 'bg-raised text-paper' : 'text-muted hover:text-paper'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {searching && <div className="text-muted text-sm">Searching…</div>}

          {results && !searching && (
            <div className="space-y-4">
              {results.length === 0 ? (
                <div className="text-muted text-sm">No results found for this query and method.</div>
              ) : (
                results.map((r, i) => (
                  <EvidenceCard key={i} page={r.page} section={r.section} text={r.text} relevance={r.relevance} />
                ))
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
