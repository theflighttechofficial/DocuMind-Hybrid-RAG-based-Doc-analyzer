import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../services/api.js'

/** Handles the common pattern: load ready documents, let the user pick one,
 * sync selection to the URL param, and hand the resolved doc up to the page. */
export default function useDocumentPicker(basePath) {
  const { id } = useParams()
  const navigate = useNavigate()
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refresh = useCallback(() => {
    setLoading(true)
    setError(null)
    return api.listDocuments()
      .then((docs) => setDocuments(docs.filter((d) => d.processing_status === 'ready')))
      .catch((err) => setError(err.message || 'Failed to load documents.'))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const selected = documents.find((d) => d.id === id) || null

  const selectDocument = (docId) => navigate(`${basePath}/${docId}`)

  return { documents, selected, selectDocument, loading, error, refresh }
}

export function DocumentPickerBar({ documents, selectedId, onSelect }) {
  if (documents.length === 0) return null
  return (
    <select
      value={selectedId || ''}
      onChange={(e) => onSelect(e.target.value)}
      className="bg-surface/70 backdrop-blur-xl border border-line/15 rounded-xl px-3 py-2 text-sm text-paper font-mono focus:border-signal outline-none"
    >
      <option value="" disabled>Select a document…</option>
      {documents.map((d) => (
        <option key={d.id} value={d.id}>{d.name}</option>
      ))}
    </select>
  )
}
