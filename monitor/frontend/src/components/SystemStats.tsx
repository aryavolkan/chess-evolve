import { useEffect, useState } from 'react'
import type { SystemInfo } from '../types'
import { getSystem } from '../api'

function Bar({ label, value, max, unit }: { label: string; value: number; max: number; unit: string }) {
  const pct = Math.min(100, (value / max) * 100)
  const color = pct < 60 ? 'bg-emerald-500' : pct < 85 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-3 flex-1">
      <span className="text-sm text-zinc-400 w-16">{label}</span>
      <div className="flex-1 h-3 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-sm text-zinc-300 w-24 text-right">
        {value.toFixed(1)}{unit} / {max.toFixed(1)}{unit}
      </span>
    </div>
  )
}

export default function SystemStats() {
  const [info, setInfo] = useState<SystemInfo | null>(null)

  useEffect(() => {
    const poll = () => getSystem().then(setInfo).catch(() => {})
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [])

  if (!info) return null

  return (
    <div className="flex gap-6 p-4 bg-zinc-900 border border-zinc-800 rounded-xl">
      <Bar label="CPU" value={info.cpu_percent} max={100} unit="%" />
      <Bar label="Memory" value={info.mem_used_gb} max={info.mem_total_gb} unit=" GB" />
      <div className="text-sm text-zinc-500 flex items-center">{info.cpu_cores} cores</div>
    </div>
  )
}
