import { useEffect, useRef, useState } from 'react'
import { getWorkerLogs } from '../api'

interface Props {
  pid: number
  onClose: () => void
}

export default function LogViewer({ pid, onClose }: Props) {
  const [log, setLog] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const poll = () => getWorkerLogs(pid, 200).then(setLog).catch(() => {})
    poll()
    const id = setInterval(poll, 2000)
    return () => clearInterval(id)
  }, [pid])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-zinc-950 border border-zinc-700 rounded-2xl w-full max-w-4xl h-[70vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
          <span className="text-sm text-zinc-300">
            Logs - <span className="font-mono text-blue-400">PID {pid}</span>
          </span>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-sm">Close</button>
        </div>
        <div className="flex-1 overflow-auto p-4">
          <pre className="text-xs font-mono text-zinc-400 whitespace-pre-wrap leading-relaxed">{log}</pre>
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  )
}
