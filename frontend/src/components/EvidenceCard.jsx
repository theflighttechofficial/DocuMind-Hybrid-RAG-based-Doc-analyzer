/** The marginalia-style evidence block. Green rule = grounded, amber = uncertain. */
export default function EvidenceCard({ page, section, text, relevance, onJump, documentName }) {
  return (
    <div className="relative pl-4 border-l-2 border-signalDim py-1">
      {documentName && (
        <div className="font-mono text-[10px] text-muted uppercase tracking-wide mb-1">
          {documentName}
        </div>
      )}
      <div className="flex items-baseline justify-between gap-3 mb-1.5">
        <div className="font-mono text-xs text-signal">
          Page {page}{section ? ` · ${section}` : ''}
        </div>
        <div className="font-mono text-xs text-muted whitespace-nowrap">
          relevance {relevance.toFixed(2)}
        </div>
      </div>
      <p className="text-sm text-paper/85 leading-relaxed">
        "{text.length > 320 ? text.slice(0, 320) + '…' : text}"
      </p>
      {onJump && (
        <button
          onClick={onJump}
          className="mt-1.5 text-xs font-mono text-signal hover:text-paper transition-colors"
        >
          view in document →
        </button>
      )}
    </div>
  )
}
