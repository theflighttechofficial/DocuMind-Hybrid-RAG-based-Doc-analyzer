import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Compass, FileText, MessageSquareText, SearchIcon, BrainCircuit,
  ScrollText, GitCompareArrows, Share2, Gauge, Menu, X, Layers, LogOut,
  Settings as SettingsIcon
} from 'lucide-react'
import { useAuth } from '../context/AuthContext.jsx'

const NAV_ITEMS = [
  { to: '/', icon: Compass, label: 'Overview', end: true },
  { to: '/documents', icon: FileText, label: 'Documents' },
  { to: '/workspaces', icon: Layers, label: 'Workspaces' },
  { to: '/ask', icon: MessageSquareText, label: 'Ask' },
  { to: '/search', icon: SearchIcon, label: 'Search' },
  { to: '/insights', icon: BrainCircuit, label: 'Insights' },
  { to: '/summarize', icon: ScrollText, label: 'Summarize' },
  { to: '/compare', icon: GitCompareArrows, label: 'Compare' },
  { to: '/graph', icon: Share2, label: 'Graph' },
  { to: '/evaluation', icon: Gauge, label: 'Evaluation' },
  { to: '/settings', icon: SettingsIcon, label: 'Settings' },
]

function NavPill({ to, icon: Icon, label, end, onNavigate }) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onNavigate}
      className={({ isActive }) =>
        `group flex items-center gap-1.5 px-3.5 py-2 rounded-full text-[13px] font-medium whitespace-nowrap transition-all ${
          isActive
            ? 'bg-signal text-white shadow-md shadow-signal/30'
            : 'text-muted hover:text-paper hover:bg-raised/70'
        }`
      }
    >
      <Icon size={14} strokeWidth={2.25} />
      <span>{label}</span>
    </NavLink>
  )
}

export default function Shell({ children }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const { user, logout } = useAuth()

  return (
    <div className="min-h-screen bg-ink text-paper font-body">
      <header className="fixed top-0 inset-x-0 z-30 px-3 sm:px-5 pt-3 sm:pt-4">
        <div className="max-w-7xl mx-auto">
          <div className="glass-nav rounded-[28px] px-3 sm:px-4 py-2 flex items-center gap-3 shadow-xl shadow-black/10">
            <div className="flex items-center gap-2 shrink-0 pl-1">
              <div className="w-8 h-8 rounded-full bg-signal flex items-center justify-center text-white font-display text-sm shadow-sm shadow-signal/40">
                D
              </div>
              <span className="font-display text-[15px] tracking-tight hidden sm:inline">DocuMind</span>
            </div>

            <nav className="hidden lg:flex items-center gap-0.5 overflow-x-auto no-scrollbar flex-1 min-w-0">
              {NAV_ITEMS.map((item) => <NavPill key={item.to} {...item} />)}
            </nav>

            <div className="flex items-center gap-1.5 shrink-0 ml-auto">
              <span
                className="hidden md:inline text-xs text-muted font-mono truncate max-w-[130px] mr-1"
                title={user?.email}
              >
                {user?.email}
              </span>
              <button
                onClick={logout}
                className="p-2 rounded-full text-muted hover:text-paper hover:bg-raised/70 transition-colors"
                title="Log out"
              >
                <LogOut size={16} />
              </button>
              <button
                onClick={() => setMenuOpen(true)}
                className="lg:hidden p-2 rounded-full text-muted hover:text-paper hover:bg-raised/70 transition-colors"
              >
                <Menu size={18} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile/tablet nav — dropdown sheet instead of a side drawer */}
      {menuOpen && (
        <div className="lg:hidden fixed inset-0 z-40">
          <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={() => setMenuOpen(false)} />
          <div className="absolute top-3 left-3 right-3 sm:left-5 sm:right-5 glass-panel rounded-[28px] p-4 animate-fade-in max-h-[75vh] overflow-y-auto shadow-2xl">
            <div className="flex items-center justify-between mb-3 px-1">
              <span className="font-display text-base">DocuMind</span>
              <button onClick={() => setMenuOpen(false)} className="p-1.5 rounded-full text-muted hover:text-paper hover:bg-raised/70">
                <X size={18} />
              </button>
            </div>
            <div className="flex flex-col gap-1">
              {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  onClick={() => setMenuOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-2.5 rounded-2xl text-sm font-medium transition-colors ${
                      isActive ? 'bg-signal text-white' : 'text-muted hover:text-paper hover:bg-raised/70'
                    }`
                  }
                >
                  <Icon size={16} strokeWidth={2.25} />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          </div>
        </div>
      )}

      <main className="pt-24 sm:pt-28 min-h-screen">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 md:px-10 pb-12">
          {children}
        </div>
      </main>
    </div>
  )
}
