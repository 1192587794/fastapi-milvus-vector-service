import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import ChatView from './views/ChatView'
import DocumentView from './views/DocumentView'
import GraphView from './views/GraphView'
import DashboardView from './views/DashboardView'

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<ChatView />} />
        <Route path="documents" element={<DocumentView />} />
        <Route path="graph" element={<GraphView />} />
        <Route path="dashboard" element={<DashboardView />} />
      </Route>
    </Routes>
  )
}

export default App
