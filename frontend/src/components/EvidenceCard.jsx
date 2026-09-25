/** The marginalia-style evidence block. Green rule = grounded, amber = uncertain. */
export default function EvidenceCard({ page, section, text, relevance, onJump, documentName, citation, cited }) {
  return (
    <div className="relative pl-4 border-l-2 border-signalDim py-1">
      {documentName && (
        <div className="font-mono text-[10px] text-muted uppercase tracking-wide mb-1">
          {documentName}
        </div>
      )}
      <div className="flex items-baseline justify-between gap-3 mb-1.5">
        <div className="font-mono text-xs text-signal flex items-baseline gap-2 min-w-0">
          {citation != null && (
            <span
              className="shrink-0 text-paper bg-signal/15 rounded px-1.5"
              title={cited ? `Cited in the answer as [${citation}]` : 'Not cited; shown because its relevance is above the threshold'}
            >
              [{citation}]
            </span>
          )}
          <span className="truncate">Page {page}{section ? ` · ${section}` : ''}</span>
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

/** Highest relevance first (the API already sends this order; kept here so
 *  older saved messages render the same way). */
export function sortByRelevance(evidence = []) {
  return [...evidence].sort((a, b) => (b.relevance ?? 0) - (a.relevance ?? 0))
}
