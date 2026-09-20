import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { MessageSquareText, SearchIcon } from 'lucide-react'
import { api } from '../services/api.js'

export default function DocumentView() {
  const { id } = useParams()
  const [doc, setDoc] = useState(null)
  const [pages, setPages] = useState([])
  const [currentPage, setCurrentPage] = useState(1)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.getDocument(id), api.getPages(id)]).then(([d, p]) => {
      setDoc(d)
      setPages(p)
      setLoading(false)
    })
  }, [id])

  if (loading) return <div className="text-muted text-sm">Loading document…</div>
  if (!doc) return <div className="text-muted text-sm">Document not found.</div>

  const pageText = pages.find((p) => p.page_number === currentPage)?.text || ''
  const toc = doc.toc_json || []
  const topKeywords = (doc.keywords_json?.tfidf || []).slice(0, 8)
  const topEntities = Object.entries(doc.entities_json || {}).flatMap(([type, items]) =>
    items.slice(0, 3).map((i) => ({ ...i, type }))
  ).slice(0, 8)

  return (
    <div className="animate-fade-in">
      <div className="flex items-start justify-between gap-4 mb-6">
        <div>
          <h1 className="font-display text-xl text-paper">{doc.name}</h1>
          <div className="text-xs text-muted font-mono mt-1">
            {doc.page_count} pages · {doc.word_count.toLocaleString()} words
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <Link to={`/ask/${id}`} className="flex items-center gap-1.5 text-xs font-mono text-ink bg-signal hover:bg-paper transition-colors px-3 py-2 rounded-xl">
            <MessageSquareText size={13} /> Ask
          </Link>
          <Link to={`/search/${id}`} className="flex items-center gap-1.5 text-xs font-mono text-paper border border-line/15 hover:border-signalDim transition-colors px-3 py-2 rounded-xl">
            <SearchIcon size={13} /> Search
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[180px_1fr_260px] gap-5">
        {/* Contents */}
        <div className="hidden lg:block">
          <div className="text-xs text-muted font-mono mb-3 uppercase tracking-wide">Contents</div>
          <div className="space-y-1">
            {toc.length === 0 && <div className="text-xs text-muted">No sections detected.</div>}
            {toc.map((section) => (
              <div key={section} className="text-sm text-muted hover:text-paper cursor-pointer truncate py-1">
                {section}
              </div>
            ))}
          </div>
          <div className="text-xs text-muted font-mono mt-6 mb-2 uppercase tracking-wide">Page</div>
          <div className="flex flex-wrap gap-1">
            {pages.slice(0, 40).map((p) => (
              <button
                key={p.page_number}
                onClick={() => setCurrentPage(p.page_number)}
                className={`w-7 h-7 text-xs font-mono rounded ${
                  currentPage === p.page_number
                    ? 'bg-signal text-ink'
                    : 'bg-raised text-muted hover:text-paper'
                }`}
              >
                {p.page_number}
              </button>
            ))}
          </div>
        </div>

        {/* Document text */}
        <div className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-6 min-h-[400px]">
          <div className="text-xs text-muted font-mono mb-4">Page {currentPage}</div>
          <div className="text-sm text-paper/90 leading-relaxed whitespace-pre-wrap">
            {pageText || 'No text on this page.'}
          </div>
          <div className="flex justify-between mt-6 pt-4 border-t border-line/15">
            <button
              disabled={currentPage <= 1}
              onClick={() => setCurrentPage((p) => p - 1)}
              className="text-xs font-mono text-muted hover:text-paper disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ← previous
            </button>
            <button
              disabled={currentPage >= pages.length}
              onClick={() => setCurrentPage((p) => p + 1)}
              className="text-xs font-mono text-muted hover:text-paper disabled:opacity-30 disabled:cursor-not-allowed"
            >
              next →
            </button>
          </div>
        </div>

        {/* Insights */}
        <div className="space-y-6">
          <div>
            <div className="text-xs text-muted font-mono mb-2 uppercase tracking-wide">Keywords</div>
            <div className="flex flex-wrap gap-1.5">
              {topKeywords.map((k) => (
                <span key={k.term} className="text-xs font-mono text-signal bg-signal/10 px-2 py-1 rounded">
                  {k.term}
                </span>
              ))}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted font-mono mb-2 uppercase tracking-wide">Entities</div>
            <div className="space-y-1.5">
              {topEntities.map((e, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-paper/80 truncate">{e.text}</span>
                  <span className="text-muted font-mono shrink-0 ml-2">{e.type}</span>
                </div>
              ))}
              {topEntities.length === 0 && <div className="text-xs text-muted">None detected.</div>}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted font-mono mb-2 uppercase tracking-wide">Statistics</div>
            <div className="space-y-1 text-xs font-mono text-muted">
              <div>{doc.stats_json?.sentence_count} sentences</div>
              <div>{doc.stats_json?.vocabulary_size} unique words</div>
              <div>avg {doc.stats_json?.avg_sentence_length} words/sentence</div>
            </div>
          </div>
          <Link to={`/insights/${id}`} className="block text-xs font-mono text-signal hover:text-paper">
            View full NLP insights →
          </Link>
        </div>
      </div>
    </div>
  )
}
