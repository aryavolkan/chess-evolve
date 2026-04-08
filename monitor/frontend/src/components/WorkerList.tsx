import { useEffect, useState, useCallback } from 'react'
import type { Worker } from '../types'
import { getWorkers } from '../api'
import WorkerCard from './WorkerCard'

interface Props {
  onViewLogs: (pid: number) => void
  onSpawn: () => void
}

export default function WorkerList({ onViewLogs, onSpawn }: Props) {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [refreshing, setRefreshing] = useState(false)
  const [lastTime, setLastTime] = useState('')
  const inflight = { current: false }

  const poll = useCallback(() => {
    if (inflight.current) return // skip if previous poll still pending
    inflight.current = true
    setRefreshing(true)
    getWorkers()
      .then((w) => {
        setWorkers(w)
        setLastTime(new Date().toLocaleTimeString())
      })
      .catch(() => {})
      .finally(() => {
        inflight.current = false
        setTimeout(() => setRefreshing(false), 200)
      })
  }, [])

  useEffect(() => {
    poll()
    const id = setInterval(poll, 3000)

    // Browsers throttle/kill timers in background tabs.
    // Re-poll immediately when the tab becomes visible again.
    const onVisible = () => {
      if (document.visibilityState === 'visible') poll()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [poll])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-zinc-100">
            Workers
            <span className="ml-2 text-sm font-normal text-zinc-500">({workers.length})</span>
          </h2>
          <div className="flex items-center gap-1.5 text-xs text-zinc-600">
            <span className={`w-1.5 h-1.5 rounded-full transition-colors ${refreshing ? 'bg-blue-400' : 'bg-zinc-700'}`} />
            {lastTime && <span>{lastTime}</span>}
          </div>
        </div>
        <button
          onClick={onSpawn}
          className="text-sm px-4 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-500 transition"
        >
          + Spawn
        </button>
      </div>

      {workers.length === 0 ? (
        <div className="text-center py-12 text-zinc-500">
          No workers running. Click Spawn to start one.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {workers.map((w) => (
            <WorkerCard
              key={w.pid}
              worker={w}
              onKill={poll}
              onViewLogs={onViewLogs}
            />
          ))}
        </div>
      )}
    </div>
  )
}
