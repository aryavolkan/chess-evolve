import type { Worker, SystemInfo, Sweep, SpawnRequest } from './types'

const BASE = ''

export async function getWorkers(): Promise<Worker[]> {
  const res = await fetch(`${BASE}/api/workers`)
  return res.json()
}

export async function killWorker(pid: number): Promise<void> {
  await fetch(`${BASE}/api/workers/kill/${pid}`, { method: 'POST' })
}

export async function spawnWorkers(req: SpawnRequest): Promise<{ spawned: { pid: number; log_file: string }[] }> {
  const res = await fetch(`${BASE}/api/workers/spawn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  return res.json()
}

export async function getWorkerLogs(pid: number, lines = 100): Promise<string> {
  const res = await fetch(`${BASE}/api/workers/${pid}/logs?lines=${lines}`)
  const data = await res.json()
  return data.log
}

export async function getSweeps(): Promise<Sweep[]> {
  const res = await fetch(`${BASE}/api/sweeps`)
  return res.json()
}

export async function getSystem(): Promise<SystemInfo> {
  const res = await fetch(`${BASE}/api/system`)
  return res.json()
}
