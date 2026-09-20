import { useCallback, useRef, useState } from 'react'
import { Upload, Check, Circle, Loader2 } from 'lucide-react'
import { api } from '../services/api.js'

const STAGES = [
  { key: 'uploaded', label: 'Uploading' },
  { key: 'extracting', label: 'Extracting text' },
  { key: 'analyzing', label: 'Running NLP analysis' },
  { key: 'embedding', label: 'Creating embeddings' },
  { key: 'indexing', label: 'Building semantic index' },
  { key: 'ready', label: 'Ready' },
]

function stageIndex(status) {
  const idx = STAGES.findIndex((s) => s.key === status)
  return idx === -1 ? 0 : idx
}

export default function UploadZone({ onUploaded }) {
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [doc, setDoc] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setError(null)
    setUploading(true)
    setDoc(null)
    try {
      const uploaded = await api.uploadDocument(file)
      setDoc(uploaded)
      poll(uploaded.id)
    } catch (err) {
      setError(err.message || 'Upload failed.')
      setUploading(false)
    }
  }, [])

  const poll = (id) => {
    const interval = setInterval(async () => {
      try {
        const updated = await api.getDocument(id)
        setDoc(updated)
        if (updated.processing_status === 'ready') {
          clearInterval(interval)
          setUploading(false)
          onUploaded?.(updated)
        } else if (updated.processing_status === 'failed') {
          clearInterval(interval)
          setUploading(false)
          setError(updated.error_message || 'Document processing failed. Please try again.')
        }
      } catch {
        clearInterval(interval)
        setUploading(false)
      }
    }, 1200)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  if (uploading || doc) {
    const currentIndex = stageIndex(doc?.processing_status)
    return (
      <div className="rounded-xl border border-line/15 bg-surface/70 backdrop-blur-xl p-6">
        <div className="text-sm text-paper mb-4 truncate">{doc?.name}</div>
        <div className="space-y-2.5">
          {STAGES.map((stage, i) => {
            const done = i < currentIndex || doc?.processing_status === 'ready'
            const active = i === currentIndex && doc?.processing_status !== 'ready'
            return (
              <div key={stage.key} className="flex items-center gap-3">
                {done ? (
                  <Check size={15} className="text-signal shrink-0" />
                ) : active ? (
                  <Loader2 size={15} className="text-signal animate-spin shrink-0" />
                ) : (
                  <Circle size={13} className="text-line shrink-0" />
                )}
                <span className={`text-sm font-mono ${done || active ? 'text-paper' : 'text-muted'}`}>
                  {stage.label}
                </span>
              </div>
            )
          })}
        </div>
        {error && <div className="mt-4 text-sm text-rust">{error}</div>}
        {doc?.processing_status === 'ready' && (
          <div className="mt-4 text-xs text-muted font-mono">
            {doc.page_count} pages · {doc.word_count.toLocaleString()} words
          </div>
        )}
      </div>
    )
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
        dragOver ? 'border-signal bg-signal/5' : 'border-line/15 hover:border-signalDim'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <Upload size={26} className="mx-auto text-muted mb-4" strokeWidth={1.5} />
      <div className="text-paper text-sm">Drop your document here</div>
      <div className="text-muted text-sm mt-1">or browse from your computer</div>
      <div className="text-muted text-xs font-mono mt-4 tracking-wide">PDF · DOCX · TXT</div>
      {error && <div className="mt-4 text-sm text-rust">{error}</div>}
    </div>
  )
}
