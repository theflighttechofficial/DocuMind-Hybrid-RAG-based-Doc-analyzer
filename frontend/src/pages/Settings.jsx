import { useEffect, useState } from 'react'
import { Sun, Moon, Monitor, CheckCircle2, XCircle, Eye, EyeOff } from 'lucide-react'
import { api } from '../services/api.js'
import { useTheme } from '../context/ThemeContext.jsx'

const THEME_OPTIONS = [
  { value: 'system', label: 'System', icon: Monitor },
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
]

export default function Settings() {
  const { theme, setTheme } = useTheme()
  const [settings, setSettings] = useState(null)
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState(null)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  const refresh = () => api.getSettings().then(setSettings)

  useEffect(() => { refresh() }, [])

  const handleSaveKey = async (e) => {
    e.preventDefault()
    if (!apiKey.trim()) return
    setSaving(true)
    setSaveMessage(null)
    try {
      await api.setGeminiKey(apiKey.trim())
      setApiKey('')
      setSaveMessage({ ok: true, text: 'API key updated.' })
      await refresh()
    } catch (err) {
      setSaveMessage({ ok: false, text: err.message || 'Failed to update key.' })
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await api.testConnection()
      setTestResult(res)
    } catch (err) {
      setTestResult({ ok: false, message: err.message || 'Test failed.' })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="animate-fade-in max-w-2xl">
      <h1 className="font-display text-2xl text-paper mb-1">Settings</h1>
      <p className="text-muted text-sm mb-8">Appearance and connection preferences.</p>

      <section className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5 mb-6">
        <div className="text-xs text-muted font-mono uppercase tracking-wide mb-4">Appearance</div>
        <div className="flex gap-2">
          {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              onClick={() => setTheme(value)}
              className={`flex-1 flex flex-col items-center gap-1.5 py-3 rounded-xl border transition-colors ${
                theme === value ? 'border-signal bg-raised text-paper' : 'border-line/15 text-muted hover:text-paper'
              }`}
            >
              <Icon size={18} strokeWidth={1.75} />
              <span className="text-xs font-mono">{label}</span>
            </button>
          ))}
        </div>
        <p className="text-xs text-muted mt-3">
          "System" follows your OS light/dark preference automatically.
        </p>
      </section>

      <section className="rounded-2xl border border-line/15 bg-surface/70 backdrop-blur-xl p-5">
        <div className="text-xs text-muted font-mono uppercase tracking-wide mb-4">Gemini API key</div>

        {settings && (
          <div className="text-sm text-paper/85 mb-4">
            {settings.gemini_api_key_set ? (
              <>Current key: <span className="font-mono text-muted">{settings.gemini_api_key_masked}</span></>
            ) : (
              <span className="text-amber">No API key configured yet.</span>
            )}
            <div className="text-xs text-muted mt-1">Model: <span className="font-mono">{settings.gemini_model}</span></div>
          </div>
        )}

        <form onSubmit={handleSaveKey} className="flex gap-2 mb-3">
          <div className="relative flex-1">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste a new Gemini API key…"
              className="w-full bg-raised border border-line/15 rounded-xl px-3 py-2.5 pr-10 text-sm text-paper placeholder:text-muted focus:border-signal outline-none"
            />
            <button
              type="button"
              onClick={() => setShowKey((s) => !s)}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted hover:text-paper"
              tabIndex={-1}
            >
              {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          </div>
          <button
            type="submit"
            disabled={saving || !apiKey.trim()}
            className="text-sm font-mono text-ink bg-signal hover:bg-paper transition-colors px-4 py-2 rounded-xl disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </form>

        {saveMessage && (
          <div className={`text-xs mb-3 ${saveMessage.ok ? 'text-signal' : 'text-rust'}`}>{saveMessage.text}</div>
        )}

        <button
          onClick={handleTest}
          disabled={testing}
          className="text-xs font-mono text-muted hover:text-paper border border-line/15 rounded-xl px-3 py-1.5 disabled:opacity-50"
        >
          {testing ? 'Testing…' : 'Test connection'}
        </button>

        {testResult && (
          <div className={`flex items-start gap-2 mt-3 text-sm ${testResult.ok ? 'text-signal' : 'text-rust'}`}>
            {testResult.ok ? <CheckCircle2 size={16} className="mt-0.5 shrink-0" /> : <XCircle size={16} className="mt-0.5 shrink-0" />}
            <span>{testResult.message}</span>
          </div>
        )}
      </section>
    </div>
  )
}
