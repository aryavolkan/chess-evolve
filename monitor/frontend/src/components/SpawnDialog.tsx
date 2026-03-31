import { useEffect, useState } from 'react'
import type { Sweep } from '../types'
import { getSweeps, spawnWorkers } from '../api'

interface Props {
  open: boolean
  onClose: () => void
}

export default function SpawnDialog({ open, onClose }: Props) {
  const [sweeps, setSweeps] = useState<Sweep[]>([])
  const [sweepId, setSweepId] = useState('')
  const [count, setCount] = useState(1)
  const [threads, setThreads] = useState(4)
  const [spawning, setSpawning] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  useEffect(() => {
    if (open) getSweeps().then(setSweeps).catch(() => {})
  }, [open])

  if (!open) return null

  const handleSpawn = async () => {
    if (!sweepId.trim()) return
    setSpawning(true)
    setResult(null)
    try {
      const res = await spawnWorkers({ sweep_id: sweepId.trim(), count, rayon_threads: threads })
      setResult(`Spawned ${res.spawned.length} worker(s): ${res.spawned.map(s => `PID ${s.pid}`).join(', ')}`)
    } catch {
      setResult('Failed to spawn')
    }
    setSpawning(false)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-6 w-full max-w-md" onClick={e => e.stopPropagation()}>
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">Spawn Workers</h3>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-zinc-400 mb-1">Sweep ID</label>
            <input
              value={sweepId}
              onChange={e => setSweepId(e.target.value)}
              placeholder="e.g. g2qo5wsr"
              className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 font-mono placeholder:text-zinc-600"
            />
            {sweeps.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {sweeps.filter(s => s.running > 0 || s.run_count < 5).slice(0, 5).map(s => (
                  <button
                    key={s.id}
                    onClick={() => setSweepId(s.id)}
                    className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:text-blue-400 font-mono"
                  >
                    {s.id}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-xs text-zinc-400 mb-1">Workers</label>
              <input
                type="number" min={1} max={8} value={count}
                onChange={e => setCount(+e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs text-zinc-400 mb-1">Rayon Threads</label>
              <input
                type="number" min={1} max={16} value={threads}
                onChange={e => setThreads(+e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200"
              />
            </div>
          </div>
        </div>

        {result && (
          <div className="mt-3 text-xs text-emerald-400 font-mono">{result}</div>
        )}

        <div className="flex gap-3 mt-5">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition text-sm"
          >
            Close
          </button>
          <button
            onClick={handleSpawn}
            disabled={spawning || !sweepId.trim()}
            className="flex-1 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-500 transition text-sm disabled:opacity-50"
          >
            {spawning ? 'Spawning...' : 'Spawn'}
          </button>
        </div>
      </div>
    </div>
  )
}
