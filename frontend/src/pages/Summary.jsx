import { useState } from 'react'
import { ScrollText } from 'lucide-react'
import { api } from '../services/api.js'
import EmptyState from '../components/EmptyState.jsx'
import useDocumentPicker, { DocumentPickerBar } from '../hooks/useDocumentPicker.jsx'

const LENGTHS = [
  { key: 'brief', label: 'Brief' },
  { key: 'standard', label: 'Standard' },
  { key: 'detailed', label: 'Detailed' },
]

function parseSummary(raw) {
  const sections = {}
  let current = null
  for (const line of raw.split('\n')) {
    const header = line.match(/^([A-Z ]{3,30}):\s*$/) || line.match(/^([A-Z ]{3,30}):(.+)$/)
    if (header) {
      current = header[1].trim()
      sections[current] = header[2]?.trim() ? [header[2].trim()] : []
    } else if (current && line.trim()) {
      sections[current].push(line.trim())
    }
  }
  return sections
}

export default function Summary() {
  const { documents, selected, selectDocument, loading } = useDocumentPicker('/summarize')
  const [length, setLength] = useState('standard')
  const [summary, setSummary] = useState(null)
  const [generating, setGenerating] = useState(false)

  const generate = async () => {
    setGenerating(true)
    setSummary(null)
    try {
      const res = await api.summarize(selected.id, length)
      setSummary(res)
    } finally {
      setGenerating(false)
    }
  }

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={ScrollText}
        title="No documents ready yet."
        description="Upload and process a document to generate a summary."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  const sections = summary?.available ? parseSummary(summary.raw) : null

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h1 className="font-display text-xl text-paper">Summarize</h1>
        <DocumentPickerBar documents={documents} selectedId={selected?.id} onSelect={selectDocument} />
      </div>

      {!selected ? (
        <EmptyState title="Select a document to summarize." />
      ) : (
        <>
          <div className="flex items-center gap-4 mb-6">
            <div className="flex gap-1">
              {LENGTHS.map((l) => (
                <button
                  key={l.key}
                  onClick={() => setLength(l.key)}
                  className={`text-xs font-mono px-3 py-1.5 rounded-xl transition-colors ${
                    length === l.key ? 'bg-raised text-paper' : 'text-muted hover:text-paper'
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
            <button
              onClick={generate}
              disabled={generating}
              className="text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl disabled:opacity-50"
            >
              {generating ? 'Generating…' : 'Generate summary'}
            </button>
          </div>

          {generating && (
            <div className="text-muted text-sm">
              Summarizing document sections, then combining into a final summary…
            </div>
          )}

          {summary && !summary.available && (
            <div className="text-rust text-sm">{summary.raw}</div>
          )}

          {sections && (
            <div className="space-y-6">
              {Object.entries(sections).map(([heading, lines]) => (
                <div key={heading}>
                  <div className="text-xs font-mono text-signal uppercase tracking-wide mb-2">{heading}</div>
                  {lines.length === 1 && !lines[0].startsWith('-') ? (
                    <p className="text-sm text-paper/90 leading-relaxed">{lines[0]}</p>
                  ) : (
                    <ul className="space-y-1.5">
                      {lines.map((line, i) => (
                        <li key={i} className="text-sm text-paper/90 leading-relaxed pl-4 relative before:content-['–'] before:absolute before:left-0 before:text-muted">
                          {line.replace(/^-\s*/, '')}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
