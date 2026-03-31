import { useEffect, useState } from 'react'
import type { Sweep } from '../types'
import { getSweeps } from '../api'

export default function SweepOverview() {
  const [sweeps, setSweeps] = useState<Sweep[]>([])

  useEffect(() => {
    const poll = () => getSweeps().then(setSweeps).catch(() => {})
    poll()
    const id = setInterval(poll, 30000)
    return () => clearInterval(id)
  }, [])

  if (sweeps.length === 0) return null

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-zinc-800">
        <h2 className="text-sm font-semibold text-zinc-300">Recent Sweeps</h2>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-zinc-500 text-xs border-b border-zinc-800">
            <th className="text-left px-4 py-2">ID</th>
            <th className="text-left px-4 py-2">Runs</th>
            <th className="text-left px-4 py-2">Active</th>
            <th className="text-left px-4 py-2">Best WR</th>
          </tr>
        </thead>
        <tbody>
          {sweeps.map(s => (
            <tr key={s.id} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
              <td className="px-4 py-2">
                <a href={s.url} target="_blank" rel="noreferrer" className="font-mono text-blue-400 hover:underline">
                  {s.id}
                </a>
              </td>
              <td className="px-4 py-2 text-zinc-400">{s.run_count}</td>
              <td className="px-4 py-2">
                {s.running > 0 ? (
                  <span className="text-emerald-400">{s.running}</span>
                ) : (
                  <span className="text-zinc-600">0</span>
                )}
              </td>
              <td className="px-4 py-2 text-zinc-300 font-mono">
                {s.best_metric != null ? `${(s.best_metric * 100).toFixed(1)}%` : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
