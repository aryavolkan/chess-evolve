import { useEffect, useState } from 'react'
import type { Worker } from '../types'
import { getWorkers } from '../api'
import WorkerCard from './WorkerCard'

interface Props {
  onViewLogs: (pid: number) => void
  onSpawn: () => void
}

export default function WorkerList({ onViewLogs, onSpawn }: Props) {
  const [workers, setWorkers] = useState<Worker[]>([])

  const poll = () => getWorkers().then(setWorkers).catch(() => {})

  useEffect(() => {
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-zinc-100">
          Workers
          <span className="ml-2 text-sm font-normal text-zinc-500">({workers.length})</span>
        </h2>
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
