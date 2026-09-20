import { useEffect, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { api } from '../services/api.js'
import UploadZone from '../components/UploadZone.jsx'
import DocumentCard from '../components/DocumentCard.jsx'
import EmptyState from '../components/EmptyState.jsx'

export default function Documents() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = () => api.listDocuments().then((docs) => { setDocuments(docs); setLoading(false) })

  useEffect(() => { refresh() }, [])

  return (
    <div className="animate-fade-in">
      <h1 className="font-display text-2xl text-paper mb-1">Documents</h1>
      <p className="text-muted text-sm mb-6">Upload and manage the documents in your workspace.</p>

      <div className="mb-8 max-w-xl">
        <UploadZone onUploaded={refresh} />
      </div>

      {loading ? (
        <div className="text-muted text-sm">Loading…</div>
      ) : documents.length === 0 ? (
        <EmptyState title="No documents yet." description="Upload your first document above." />
      ) : (
        <div className="grid md:grid-cols-2 gap-3">
          {documents.map((doc) => (
            <DocumentRow key={doc.id} doc={doc} onDeleted={refresh} />
          ))}
        </div>
      )}
    </div>
  )
}

function DocumentRow({ doc, onDeleted }) {
  const [deleting, setDeleting] = useState(false)

  const handleDelete = async (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!confirm(`Delete "${doc.name}"? This can't be undone.`)) return
    setDeleting(true)
    await api.deleteDocument(doc.id)
    onDeleted()
  }

  return (
    <div className="relative group">
      <DocumentCard doc={doc} />
      <button
        onClick={handleDelete}
        disabled={deleting}
        className="absolute top-4 right-4 p-1.5 rounded text-muted opacity-0 group-hover:opacity-100 hover:text-rust hover:bg-ink/40 transition-all"
        title="Delete document"
      >
        <Trash2 size={14} />
      </button>
    </div>
  )
}
