import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Upload, MessageSquareText, BrainCircuit, GitCompareArrows } from 'lucide-react'
import { api } from '../services/api.js'
import DocumentCard from '../components/DocumentCard.jsx'
import EmptyState from '../components/EmptyState.jsx'

const QUICK_ACTIONS = [
  { to: '/documents', icon: Upload, label: 'Upload document' },
  { to: '/ask', icon: MessageSquareText, label: 'Ask a document' },
  { to: '/insights', icon: BrainCircuit, label: 'Analyze document' },
  { to: '/compare', icon: GitCompareArrows, label: 'Compare documents' },
]

export default function Overview() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.listDocuments().then((docs) => {
      setDocuments(docs)
      setLoading(false)
    })
  }, [])

  const readyDocs = documents.filter((d) => d.processing_status === 'ready')
  const totalPages = readyDocs.reduce((sum, d) => sum + d.page_count, 0)
  const totalWords = readyDocs.reduce((sum, d) => sum + d.word_count, 0)

  return (
    <div className="animate-fade-in">
      <div className="mb-10">
        <h1 className="font-display text-3xl md:text-4xl text-paper leading-tight">
          Good {timeOfDay()}.
        </h1>
        <p className="text-muted mt-3 max-w-lg leading-relaxed">
          Turn your documents into knowledge. Upload a document, explore its structure,
          search by meaning, and ask questions using natural language.
        </p>
        <Link
          to="/documents"
          className="inline-block mt-5 text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2.5 rounded-xl"
        >
          Upload document
        </Link>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-10">
        <Stat label="Documents" value={readyDocs.length} />
        <Stat label="Pages" value={totalPages.toLocaleString()} />
        <Stat label="Words" value={totalWords.toLocaleString()} />
        <Stat label="Questions" value="—" hint="tracked per session" />
      </div>

      <div className="mb-10">
        <h2 className="text-sm text-muted font-mono mb-3">Quick actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {QUICK_ACTIONS.map(({ to, icon: Icon, label }) => (
            <Link
              key={to}
              to={to}
              className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-4 hover:border-signalDim transition-colors flex flex-col gap-3"
            >
              <Icon size={18} className="text-signal" strokeWidth={1.5} />
              <span className="text-sm text-paper">{label}</span>
            </Link>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-sm text-muted font-mono mb-3">Recent documents</h2>
        {loading ? (
          <div className="text-muted text-sm">Loading…</div>
        ) : documents.length === 0 ? (
          <EmptyState
            title="No documents yet."
            description="Upload your first document and start exploring its contents."
            actionLabel="Upload document"
            actionTo="/documents"
          />
        ) : (
          <div className="grid md:grid-cols-2 gap-3">
            {documents.slice(0, 6).map((doc) => (
              <DocumentCard key={doc.id} doc={doc} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Stat({ label, value, hint }) {
  return (
    <div className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-4">
      <div className="font-display text-2xl text-paper">{value}</div>
      <div className="text-xs text-muted mt-1">{label}</div>
      {hint && <div className="text-[10px] text-muted/60 font-mono mt-0.5">{hint}</div>}
    </div>
  )
}

function timeOfDay() {
  const h = new Date().getHours()
  if (h < 12) return 'morning'
  if (h < 18) return 'afternoon'
  return 'evening'
}
