import { useState } from 'react'
import { Gauge } from 'lucide-react'
import { api } from '../services/api.js'
import EmptyState from '../components/EmptyState.jsx'
import useDocumentPicker, { DocumentPickerBar } from '../hooks/useDocumentPicker.jsx'

const SAMPLE_QUESTION = { question: '', answer: '', page: '' }

export default function Evaluation() {
  const { documents, selected, selectDocument, loading } = useDocumentPicker('/evaluation')
  const [evalSet, setEvalSet] = useState([{ ...SAMPLE_QUESTION }])
  const [result, setResult] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [compareRerank, setCompareRerank] = useState(false)

  const updateItem = (i, field, value) => {
    setEvalSet((prev) => prev.map((item, idx) => (idx === i ? { ...item, [field]: value } : item)))
  }

  const addItem = () => setEvalSet((prev) => [...prev, { ...SAMPLE_QUESTION }])
  const removeItem = (i) => setEvalSet((prev) => prev.filter((_, idx) => idx !== i))

  const run = async () => {
    const cleaned = evalSet
      .filter((item) => item.question.trim())
      .map((item) => ({
        question: item.question.trim(),
        answer: item.answer.trim() || null,
        page: item.page ? parseInt(item.page, 10) : null,
      }))
    if (cleaned.length === 0 || !selected) return
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.runEvaluation(selected.id, cleaned, compareRerank)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(false)
    }
  }

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={Gauge}
        title="No documents ready yet."
        description="Upload and process a document to run evaluations against it."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h1 className="font-display text-xl text-paper">Evaluation</h1>
        <DocumentPickerBar documents={documents} selectedId={selected?.id} onSelect={selectDocument} />
      </div>

      {!selected ? (
        <EmptyState title="Select a document to evaluate against." />
      ) : (
        <>
          <div className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5 mb-6">
            <div className="text-xs text-muted font-mono uppercase tracking-wide mb-4">
              Evaluation set — question, expected answer (optional), source page (optional)
            </div>
            <div className="space-y-3">
              {evalSet.map((item, i) => (
                <div key={i} className="grid grid-cols-[1fr_1fr_80px_28px] gap-2">
                  <input
                    value={item.question}
                    onChange={(e) => updateItem(i, 'question', e.target.value)}
                    placeholder="Question"
                    className="bg-raised border border-line/15 rounded px-3 py-2 text-sm text-paper placeholder:text-muted"
                  />
                  <input
                    value={item.answer}
                    onChange={(e) => updateItem(i, 'answer', e.target.value)}
                    placeholder="Expected answer (optional)"
                    className="bg-raised border border-line/15 rounded px-3 py-2 text-sm text-paper placeholder:text-muted"
                  />
                  <input
                    value={item.page}
                    onChange={(e) => updateItem(i, 'page', e.target.value)}
                    placeholder="Page"
                    className="bg-raised border border-line/15 rounded px-3 py-2 text-sm text-paper placeholder:text-muted"
                  />
                  <button onClick={() => removeItem(i)} className="text-muted hover:text-rust text-sm">×</button>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-3 mt-4">
              <button onClick={addItem} className="text-xs font-mono text-muted hover:text-paper">+ add question</button>
              <label className="flex items-center gap-1.5 text-xs font-mono text-muted cursor-pointer">
                <input
                  type="checkbox"
                  checked={compareRerank}
                  onChange={(e) => setCompareRerank(e.target.checked)}
                  className="accent-signal"
                />
                compare reranked hybrid
              </label>
              <button
                onClick={run}
                disabled={running}
                className="ml-auto text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl disabled:opacity-50"
              >
                {running ? 'Running…' : 'Run evaluation'}
              </button>
            </div>
          </div>

          {error && <div className="text-rust text-sm mb-4">{error}</div>}

          {result && (
            <div className="grid md:grid-cols-2 gap-6">
              <Section title="Retrieval comparison">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-muted font-mono">
                      <th className="text-left font-normal py-1">Method</th>
                      <th className="text-right font-normal py-1">P@5</th>
                      <th className="text-right font-normal py-1">R@5</th>
                      <th className="text-right font-normal py-1">MRR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.retrieval).map(([method, m]) => (
                      <tr key={method} className="border-t border-line/15">
                        <td className="py-2 text-paper capitalize">{method.replace('_', ' + ')}</td>
                        <td className="py-2 text-right font-mono text-paper">{m.precision ?? '—'}</td>
                        <td className="py-2 text-right font-mono text-paper">{m.recall ?? '—'}</td>
                        <td className="py-2 text-right font-mono text-paper">{m.mrr ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              <Section title="Question answering">
                <div className="grid grid-cols-2 gap-4">
                  <Metric label="Exact Match" value={result.qa.exact_match} />
                  <Metric label="F1" value={result.qa.f1} />
                  <Metric label="Semantic Sim." value={result.qa.semantic_similarity} />
                  <Metric label="BERTScore F1" value={result.qa.bertscore_f1} />
                </div>
              </Section>

              {result.summarization && (
                <Section title="Summarization">
                  <div className="grid grid-cols-3 gap-4">
                    <Metric label="ROUGE-1" value={result.summarization.rouge1} />
                    <Metric label="ROUGE-2" value={result.summarization.rouge2} />
                    <Metric label="ROUGE-L" value={result.summarization.rougeL} />
                  </div>
                </Section>
              )}

              <Section title="Sample size">
                <div className="font-display text-2xl text-paper">{result.sample_size}</div>
                <div className="text-xs text-muted mt-1">questions evaluated</div>
              </Section>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5">
      <div className="text-xs text-muted font-mono uppercase tracking-wide mb-4">{title}</div>
      {children}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div>
      <div className="font-display text-xl text-paper">{value ?? '—'}</div>
      <div className="text-xs text-muted mt-0.5">{label}</div>
    </div>
  )
}
