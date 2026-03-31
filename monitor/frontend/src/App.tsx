import { useState } from 'react'
import SystemStats from './components/SystemStats'
import SweepOverview from './components/SweepOverview'
import WorkerList from './components/WorkerList'
import SpawnDialog from './components/SpawnDialog'
import LogViewer from './components/LogViewer'

export default function App() {
  const [showSpawn, setShowSpawn] = useState(false)
  const [logPid, setLogPid] = useState<number | null>(null)

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold tracking-tight">Chess-Evolve Monitor</h1>

        <SystemStats />
        <SweepOverview />
        <WorkerList onViewLogs={setLogPid} onSpawn={() => setShowSpawn(true)} />

        <SpawnDialog open={showSpawn} onClose={() => setShowSpawn(false)} />
        {logPid && <LogViewer pid={logPid} onClose={() => setLogPid(null)} />}
      </div>
    </div>
  )
}
