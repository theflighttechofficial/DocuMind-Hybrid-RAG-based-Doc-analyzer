import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api, setAuthToken, clearAuthToken, setUnauthorizedHandler } from '../services/api.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  // Token/user live only in React state (backed by the in-memory module
  // variable in api.js) — never persisted, so a page refresh always requires
  // logging in again. This is deliberate: it avoids localStorage XSS token theft.
  const [user, setUser] = useState(null)

  const logout = useCallback(() => {
    clearAuthToken()
    setUser(null)
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(logout)
    return () => setUnauthorizedHandler(null)
  }, [logout])

  const login = async (email, password) => {
    const { access_token } = await api.login(email, password)
    setAuthToken(access_token)
    const me = await api.me()
    setUser(me)
  }

  const register = async (email, password) => {
    const { access_token } = await api.register(email, password)
    setAuthToken(access_token)
    const me = await api.me()
    setUser(me)
  }

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
