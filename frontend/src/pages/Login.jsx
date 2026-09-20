import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useAuth } from '../context/AuthContext.jsx'

export default function Login() {
  const { login, register } = useAuth()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState(searchParams.get('mode') === 'register' ? 'register' : 'login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      if (mode === 'login') {
        await login(email.trim(), password)
      } else {
        await register(email.trim(), password)
      }
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center px-5 py-10">
      <div className="w-full max-w-sm animate-fade-in">
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 text-xs font-mono text-muted hover:text-paper transition-colors mb-8"
        >
          <ArrowLeft size={13} /> Back home
        </Link>

        <div className="mb-8 text-center">
          <div className="font-display text-2xl text-paper">DocuMind</div>
          <div className="text-xs text-muted font-mono mt-1.5">evidence-first documents</div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: 'easeOut' }}
          className="rounded-[28px] glass-panel shadow-xl shadow-black/10 p-6"
        >
          <div className="relative flex gap-1 mb-6 p-1 rounded-full bg-raised/70 border border-line/15 text-sm font-mono">
            {['login', 'register'].map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => { setMode(m); setError(null) }}
                className="relative flex-1 py-2 rounded-full transition-colors z-0"
                style={{ color: mode === m ? 'rgb(var(--color-ink))' : undefined }}
              >
                {mode === m && (
                  <motion.div
                    layoutId="login-tab-pill"
                    transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                    className="absolute inset-0 rounded-full bg-signal -z-10"
                  />
                )}
                <span className={mode === m ? '' : 'text-muted'}>
                  {m === 'login' ? 'Log in' : 'Register'}
                </span>
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            <motion.form
              key={mode}
              onSubmit={handleSubmit}
              initial={{ opacity: 0, x: mode === 'login' ? -8 : 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: mode === 'login' ? 8 : -8 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
              className="space-y-3"
            >
              <div>
                <label className="block text-xs text-muted font-mono mb-1.5">Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-raised/70 border border-line/15 rounded-2xl px-3.5 py-2.5 text-sm text-paper placeholder:text-muted focus:border-signal outline-none"
                  placeholder="you@example.com"
                />
              </div>
              <div>
                <label className="block text-xs text-muted font-mono mb-1.5">Password</label>
                <input
                  type="password"
                  required
                  minLength={mode === 'register' ? 8 : undefined}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-raised/70 border border-line/15 rounded-2xl px-3.5 py-2.5 text-sm text-paper placeholder:text-muted focus:border-signal outline-none"
                  placeholder={mode === 'register' ? 'At least 8 characters' : '••••••••'}
                />
              </div>

              {error && <div className="text-rust text-xs">{error}</div>}

              <motion.button
                type="submit"
                disabled={submitting}
                whileHover={{ scale: submitting ? 1 : 1.01 }}
                whileTap={{ scale: submitting ? 1 : 0.98 }}
                className="w-full mt-2 text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2.5 rounded-2xl disabled:opacity-50"
              >
                {submitting ? 'Please wait…' : mode === 'login' ? 'Log in' : 'Create account'}
              </motion.button>
            </motion.form>
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  )
}
