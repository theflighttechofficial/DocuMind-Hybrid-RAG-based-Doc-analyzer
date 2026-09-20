import { Routes, Route, Navigate } from 'react-router-dom'
import Shell from './components/Shell.jsx'
import Landing from './pages/Landing.jsx'
import Login from './pages/Login.jsx'
import { useAuth } from './context/AuthContext.jsx'
import Overview from './pages/Overview.jsx'
import Documents from './pages/Documents.jsx'
import DocumentView from './pages/DocumentView.jsx'
import Workspaces from './pages/Workspaces.jsx'
import Ask from './pages/Ask.jsx'
import Search from './pages/Search.jsx'
import Insights from './pages/Insights.jsx'
import Summary from './pages/Summary.jsx'
import Compare from './pages/Compare.jsx'
import KnowledgeGraph from './pages/KnowledgeGraph.jsx'
import Evaluation from './pages/Evaluation.jsx'
import Settings from './pages/Settings.jsx'

export default function App() {
  const { user } = useAuth()

  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    )
  }

  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/login" element={<Navigate to="/" replace />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/documents/:id" element={<DocumentView />} />
        <Route path="/workspaces" element={<Workspaces />} />
        <Route path="/ask" element={<Ask />} />
        <Route path="/ask/:id" element={<Ask />} />
        <Route path="/search" element={<Search />} />
        <Route path="/search/:id" element={<Search />} />
        <Route path="/insights" element={<Insights />} />
        <Route path="/insights/:id" element={<Insights />} />
        <Route path="/summarize" element={<Summary />} />
        <Route path="/summarize/:id" element={<Summary />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/graph" element={<KnowledgeGraph />} />
        <Route path="/graph/:id" element={<KnowledgeGraph />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/evaluation/:id" element={<Evaluation />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Shell>
  )
}
