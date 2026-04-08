import type { Worker } from '../types'
import { killWorker } from '../api'

function formatUptime(sec: number): string {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

interface Props {
  worker: Worker
  onKill: () => void
  onViewLogs: (pid: number) => void
}

export default function WorkerCard({ worker, onKill, onViewLogs }: Props) {
  const statusColor = worker.status === 'running' ? 'bg-emerald-500' : 'bg-amber-500'
  const cpuColor = worker.cpu_pct > 200 ? 'text-emerald-400' : worker.cpu_pct > 5 ? 'text-zinc-300' : 'text-zinc-500'

  const handleKill = async () => {
    if (!confirm(`Kill worker PID ${worker.pid}?`)) return
    await killWorker(worker.pid)
    onKill()
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="font-mono text-sm text-zinc-200">PID {worker.pid}</span>
        </div>
        <span className="text-xs text-zinc-500">{formatUptime(worker.uptime_sec)}</span>
      </div>

      {worker.sweep_id && (
        <div className="text-xs text-zinc-400">
          Sweep: <span className="font-mono text-blue-400">{worker.sweep_id}</span>
        </div>
      )}

      <div className="flex gap-4 text-xs">
        <span className={cpuColor}>CPU {worker.cpu_pct.toFixed(0)}%</span>
        <span className="text-zinc-400">MEM {worker.mem_pct.toFixed(1)}%</span>
        {worker.generation != null && (
          <span className="text-zinc-300">Gen {worker.generation}</span>
        )}
        {worker.run_count > 0 && (
          <span className="text-zinc-500">Run #{worker.run_count}</span>
        )}
      </div>

      {worker.last_log_line && (
        <div className="text-xs font-mono text-zinc-600 truncate" title={worker.last_log_line}>
          {worker.last_log_line}
        </div>
      )}

      <div className="flex gap-2 mt-auto pt-2">
        <button
          onClick={() => onViewLogs(worker.pid)}
          className="flex-1 text-xs py-1.5 rounded-lg bg-zinc-800 text-zinc-300 hover:bg-zinc-700 transition"
        >
          Logs
        </button>
        <button
          onClick={handleKill}
          className="flex-1 text-xs py-1.5 rounded-lg bg-red-950 text-red-400 hover:bg-red-900 transition"
        >
          Kill
        </button>
      </div>
    </div>
  )
}
