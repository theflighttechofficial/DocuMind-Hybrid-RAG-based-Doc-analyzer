import { Link } from 'react-router-dom'
import { FileText, Loader2, AlertTriangle } from 'lucide-react'

const STATUS_LABEL = {
  uploaded: 'Queued',
  extracting: 'Extracting text',
  analyzing: 'Running NLP analysis',
  embedding: 'Creating embeddings',
  indexing: 'Building semantic index',
  ready: 'Ready',
  failed: 'Failed',
}

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export default function DocumentCard({ doc }) {
  const isReady = doc.processing_status === 'ready'
  const isFailed = doc.processing_status === 'failed'
  const isProcessing = !isReady && !isFailed

  const content = (
    <div className="group rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-4 hover:border-signalDim transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0">
          <FileText size={18} className="text-muted mt-0.5 shrink-0" strokeWidth={1.5} />
          <div className="min-w-0">
            <div className="text-sm text-paper truncate">{doc.name}</div>
            <div className="text-xs text-muted font-mono mt-1">
              {doc.page_count > 0 ? `${doc.page_count} pages · ` : ''}
              {doc.word_count > 0 ? `${(doc.word_count / 1000).toFixed(1)}k words · ` : ''}
              {timeAgo(doc.created_at)}
            </div>
          </div>
        </div>
        {isProcessing && <Loader2 size={16} className="text-signal animate-spin shrink-0" />}
        {isFailed && <AlertTriangle size={16} className="text-rust shrink-0" />}
      </div>
      {!isReady && (
        <div className={`mt-3 text-xs font-mono ${isFailed ? 'text-rust' : 'text-muted'}`}>
          {STATUS_LABEL[doc.processing_status] || doc.processing_status}
        </div>
      )}
    </div>
  )

  return isReady ? <Link to={`/documents/${doc.id}`}>{content}</Link> : content
}
