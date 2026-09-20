import { useEffect, useMemo, useState } from 'react'
import ReactFlow, { Background, Controls, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'
import { Share2, RotateCcw } from 'lucide-react'
import { api } from '../services/api.js'
import EmptyState from '../components/EmptyState.jsx'
import useDocumentPicker, { DocumentPickerBar } from '../hooks/useDocumentPicker.jsx'

const TYPE_COLOR = {
  PERSON: '#0A84FF',
  ORGANIZATION: '#FF9F0A',
  LOCATION: '#FF453A',
  DATE: '#8E8E93',
}

function layoutNodes(nodes) {
  // Concentric rings instead of one ever-widening circle — a single-circle
  // layout's radius (and thus required canvas size) grows linearly with node
  // count, which is what was pushing large graphs past what fitView could
  // zoom out to (see minZoom below) and cutting off the outer nodes.
  // Rings keep the overall footprint growing roughly with sqrt(n) instead.
  const perRing = 14
  const ringGap = 170
  return nodes.map((n, i) => {
    const ring = Math.floor(i / perRing)
    const indexInRing = i % perRing
    const countInRing = Math.min(perRing, nodes.length - ring * perRing)
    const radius = 140 + ring * ringGap
    const angle = (2 * Math.PI * indexInRing) / countInRing
    return {
      id: n.id,
      position: { x: radius * Math.cos(angle), y: radius * Math.sin(angle) },
      data: { label: n.label, mentions: n.mentions, pages: n.pages, type: n.type },
      style: {
        background: '#181818',
        border: `1.5px solid ${TYPE_COLOR[n.type] || '#3A3A3C'}`,
        borderRadius: 14,
        color: '#F5F5F7',
        fontSize: 12,
        fontFamily: 'IBM Plex Mono, monospace',
        padding: '6px 10px',
      },
    }
  })
}

export default function KnowledgeGraph() {
  const { documents, selected, selectDocument, loading, error: listError, refresh } = useDocumentPicker('/graph')
  const [graph, setGraph] = useState(null)
  const [graphError, setGraphError] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const [retryTick, setRetryTick] = useState(0)

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    setGraph(null)
    setGraphError(null)
    setSelectedNode(null)
    api.getGraph(selected.id)
      .then((g) => { if (!cancelled) setGraph(g) })
      .catch((err) => { if (!cancelled) setGraphError(err.message || 'Failed to build the knowledge graph.') })
    // if the user switches documents before this resolves, ignore the stale response
    // instead of rendering document A's graph under document B's selection
    return () => { cancelled = true }
  }, [selected?.id, retryTick])

  const flowNodes = useMemo(() => (graph ? layoutNodes(graph.nodes) : []), [graph])
  const flowEdges = useMemo(
    () =>
      graph
        ? graph.edges.map((e, i) => ({
            id: `e${i}`,
            source: e.source,
            target: e.target,
            style: { stroke: '#48484A', strokeWidth: Math.min(1 + e.weight * 0.5, 4) },
            markerEnd: { type: MarkerType.ArrowClosed, color: '#48484A' },
          }))
        : [],
    [graph]
  )

  if (loading) return <div className="text-muted text-sm">Loading…</div>

  if (listError) {
    return (
      <EmptyState icon={Share2} title="Couldn't load your documents." description={listError}>
        <button
          onClick={refresh}
          className="inline-flex items-center gap-1.5 mt-5 text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl"
        >
          <RotateCcw size={14} /> Retry
        </button>
      </EmptyState>
    )
  }

  if (documents.length === 0) {
    return (
      <EmptyState
        icon={Share2}
        title="No documents ready yet."
        description="Upload and process a document to explore its knowledge graph."
        actionLabel="Upload document"
        actionTo="/documents"
      />
    )
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h1 className="font-display text-xl text-paper">Knowledge Graph</h1>
        <DocumentPickerBar documents={documents} selectedId={selected?.id} onSelect={selectDocument} />
      </div>

      {!selected ? (
        <EmptyState title="Select a document to explore." />
      ) : graphError ? (
        <EmptyState title="Couldn't build the knowledge graph." description={graphError}>
          <button
            onClick={() => setRetryTick((t) => t + 1)}
            className="inline-flex items-center gap-1.5 mt-5 text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl"
          >
            <RotateCcw size={14} /> Retry
          </button>
        </EmptyState>
      ) : !graph ? (
        <div className="text-muted text-sm">Building graph…</div>
      ) : graph.nodes.length === 0 ? (
        <EmptyState title="Not enough entities found to build a graph." />
      ) : (
        <div className="grid lg:grid-cols-[1fr_260px] gap-4">
          <div
            className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl"
            style={{ height: Math.max(520, 360 + Math.ceil(graph.nodes.length / 14) * 90) }}
          >
            {/* key forces a clean remount per document/graph so fitView always
                measures the actual node set instead of a stale viewport */}
            <ReactFlow
              key={selected.id}
              nodes={flowNodes}
              edges={flowEdges}
              onNodeClick={(_, node) => setSelectedNode(graph.nodes.find((n) => n.id === node.id))}
              fitView
              fitViewOptions={{ padding: 0.25 }}
              minZoom={0.05}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="#3A3A3C" gap={20} />
              <Controls showInteractive={false} />
            </ReactFlow>
          </div>
          <div className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5">
            <div className="text-xs text-muted font-mono uppercase tracking-wide mb-3">Entity</div>
            {selectedNode ? (
              <div>
                <div className="font-display text-lg text-paper mb-1">{selectedNode.label}</div>
                <div className="text-xs font-mono mb-4" style={{ color: TYPE_COLOR[selectedNode.type] }}>
                  {selectedNode.type}
                </div>
                <div className="text-xs text-muted mb-1">Mentions</div>
                <div className="text-sm text-paper mb-4">{selectedNode.mentions}</div>
                <div className="text-xs text-muted mb-1">Source pages</div>
                <div className="text-sm text-paper font-mono">{selectedNode.pages.join(', ') || '—'}</div>
              </div>
            ) : (
              <div className="text-sm text-muted">Click a node to see details.</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
