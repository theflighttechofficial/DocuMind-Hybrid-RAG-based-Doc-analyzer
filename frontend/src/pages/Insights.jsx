import { useEffect, useState } from 'react'
import { BrainCircuit } from 'lucide-react'
import { api } from '../services/api.js'
import EmptyState from '../components/EmptyState.jsx'
import useDocumentPicker, { DocumentPickerBar } from '../hooks/useDocumentPicker.jsx'

const ENTITY_COLORS = {
  PERSON: 'text-signal bg-signal/10',
  ORGANIZATION: 'text-amber bg-amber/10',
  LOCATION: 'text-rust bg-rust/10',
  DATE: 'text-muted bg-raised',
}

export default function Insights() {
  const { documents, selected, selectDocument, loading } = useDocumentPicker('/insights')
  const [data, setData] = useState(null)

  useEffect(() => {
    if (!selected) return
    setData(null)
    Promise.all([
      api.getEntities(selected.id),
      api.getKeywords(selected.id),
      api.getPos(selected.id),
      api.getStatistics(selected.id),
    ]).then(([entities, keywords, pos, stats]) => setData({ entities, keywords, pos, stats }))
  }, [selected?.id])

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={BrainCircuit}
        title="No documents ready yet."
        description="Upload and process a document to see its NLP insights."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h1 className="font-display text-xl text-paper">NLP Insights</h1>
        <DocumentPickerBar documents={documents} selectedId={selected?.id} onSelect={selectDocument} />
      </div>

      {!selected ? (
        <EmptyState title="Select a document to analyze." />
      ) : !data ? (
        <div className="text-muted text-sm">Loading analysis…</div>
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          <Section title="Named Entities">
            <div className="space-y-4">
              {Object.entries(data.entities).length === 0 && (
                <div className="text-sm text-muted">No entities detected.</div>
              )}
              {Object.entries(data.entities).map(([type, items]) => (
                <div key={type}>
                  <div className="text-xs font-mono text-muted uppercase tracking-wide mb-1.5">{type}</div>
                  <div className="flex flex-wrap gap-1.5">
                    {items.slice(0, 12).map((item, i) => (
                      <span key={i} className={`text-xs px-2 py-1 rounded font-mono ${ENTITY_COLORS[type] || 'text-muted bg-raised'}`}>
                        {item.text} <span className="opacity-60">×{item.count}</span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Top Keywords (TF-IDF)">
            <BarList items={data.keywords.tfidf?.slice(0, 10) || []} labelKey="term" valueKey="score" />
          </Section>

          <Section title="Top Keywords (TextRank)">
            <BarList items={data.keywords.textrank?.slice(0, 10) || []} labelKey="term" valueKey="score" />
          </Section>

          <Section title="Top Keywords (KeyBERT)">
            <BarList items={data.keywords.keybert?.slice(0, 10) || []} labelKey="term" valueKey="score" />
          </Section>

          <Section title="Part-of-Speech Distribution">
            <div className="space-y-2">
              {Object.entries(data.pos).map(([tag, pct]) => (
                <div key={tag} className="flex items-center gap-3">
                  <div className="text-xs font-mono text-muted w-28 shrink-0">{tag}</div>
                  <div className="flex-1 h-1.5 bg-raised rounded-full overflow-hidden">
                    <div className="h-full bg-signal rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="text-xs font-mono text-paper w-12 text-right">{pct}%</div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Document Statistics">
            <div className="grid grid-cols-2 gap-4">
              <Stat label="Words" value={data.stats.word_count?.toLocaleString()} />
              <Stat label="Sentences" value={data.stats.sentence_count?.toLocaleString()} />
              <Stat label="Paragraphs" value={data.stats.paragraph_count?.toLocaleString()} />
              <Stat label="Avg sentence length" value={`${data.stats.avg_sentence_length} words`} />
              <Stat label="Vocabulary size" value={data.stats.vocabulary_size?.toLocaleString()} />
              <Stat label="Unique word ratio" value={data.stats.unique_word_ratio} />
              <Stat label="Flesch reading ease" value={data.stats.flesch_reading_ease} />
            </div>
          </Section>
        </div>
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

function BarList({ items, labelKey, valueKey }) {
  if (items.length === 0) return <div className="text-sm text-muted">Not enough text to extract keywords.</div>
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="text-xs font-mono text-paper w-28 shrink-0 truncate">{item[labelKey]}</div>
          <div className="flex-1 h-1.5 bg-raised rounded-full overflow-hidden">
            <div className="h-full bg-signal rounded-full" style={{ width: `${item[valueKey] * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="font-display text-xl text-paper">{value ?? '—'}</div>
      <div className="text-xs text-muted mt-0.5">{label}</div>
    </div>
  )
}
