import { Link } from 'react-router-dom'

export default function EmptyState({ icon: Icon, title, description, actionLabel, actionTo, children }) {
  return (
    <div className="rounded-xl border border-line/15 bg-surface/70 backdrop-blur-xl py-16 px-6 text-center">
      {Icon && <Icon size={28} className="mx-auto text-muted mb-4" strokeWidth={1.5} />}
      <div className="text-paper text-sm">{title}</div>
      {description && <div className="text-muted text-sm mt-1.5 max-w-sm mx-auto">{description}</div>}
      {actionLabel && actionTo && (
        <Link
          to={actionTo}
          className="inline-block mt-5 text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl"
        >
          {actionLabel}
        </Link>
      )}
      {children}
    </div>
  )
}
