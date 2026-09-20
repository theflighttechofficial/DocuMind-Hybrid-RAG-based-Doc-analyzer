import { useEffect, useState } from 'react'
import { GitCompareArrows } from 'lucide-react'
import { api } from '../services/api.js'
import EmptyState from '../components/EmptyState.jsx'

const BADGE = {
  UNCHANGED: 'text-muted bg-raised',
  MODIFIED: 'text-amber bg-amber/10',
  ADDED: 'text-signal bg-signal/10',
  REMOVED: 'text-rust bg-rust/10',
}

export default function Compare() {
  const [documents, setDocuments] = useState([])
  const [idA, setIdA] = useState('')
  const [idB, setIdB] = useState('')
  const [result, setResult] = useState(null)
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.listDocuments().then((docs) => setDocuments(docs.filter((d) => d.processing_status === 'ready')))
  }, [])

  const runCompare = async () => {
    if (!idA || !idB || idA === idB) return
    setComparing(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.compareDocuments(idA, idB)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setComparing(false)
    }
  }

  if (documents.length < 2) {
    return (
      <EmptyState
        icon={GitCompareArrows}
        title="Need at least two processed documents to compare."
        description="Upload a second document to enable comparison."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  return (
    <div className="animate-fade-in">
      <h1 className="font-display text-xl text-paper mb-6">Compare</h1>

      <div className="flex flex-col md:flex-row items-stretch md:items-center gap-3 mb-8">
        <select value={idA} onChange={(e) => setIdA(e.target.value)} className="flex-1 bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-3 py-2.5 text-sm text-paper font-mono">
          <option value="" disabled>Document A</option>
          {documents.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <div className="text-muted text-sm text-center font-mono">vs</div>
        <select value={idB} onChange={(e) => setIdB(e.target.value)} className="flex-1 bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-3 py-2.5 text-sm text-paper font-mono">
          <option value="" disabled>Document B</option>
          {documents.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <button
          onClick={runCompare}
          disabled={!idA || !idB || idA === idB || comparing}
          className="text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-5 py-2.5 rounded-xl disabled:opacity-40"
        >
          {comparing ? 'Comparing…' : 'Compare'}
        </button>
      </div>

      {error && <div className="text-rust text-sm mb-4">{error}</div>}

      {result && (
        <>
          <div className="grid grid-cols-4 gap-3 mb-6">
            {Object.entries(result.summary).map(([key, count]) => (
              <div key={key} className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-4 text-center">
                <div className="font-display text-2xl text-paper">{count}</div>
                <div className={`text-xs font-mono mt-1 uppercase tracking-wide inline-block px-2 py-0.5 rounded ${BADGE[key.toUpperCase()]}`}>
                  {key}
                </div>
              </div>
            ))}
          </div>

          <div className="space-y-4">
            {result.results.map((r, i) => (
              <div key={i} className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm text-paper">{r.section || 'Untitled section'}</div>
                  <span className={`text-xs font-mono px-2 py-0.5 rounded ${BADGE[r.classification]}`}>
                    {r.classification}
                  </span>
                </div>
                <p className="text-sm text-muted mb-3">{r.explanation}</p>
                <div className="grid md:grid-cols-2 gap-4">
                  {r.version_a && (
                    <div className="border-l-2 border-line/15 pl-3">
                      <div className="text-[11px] font-mono text-muted mb-1">A · Page {r.version_a.page}</div>
                      <div className="text-xs text-paper/70">{r.version_a.text}</div>
                    </div>
                  )}
                  {r.version_b && (
                    <div className="border-l-2 border-line/15 pl-3">
                      <div className="text-[11px] font-mono text-muted mb-1">B · Page {r.version_b.page}</div>
                      <div className="text-xs text-paper/70">{r.version_b.text}</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
