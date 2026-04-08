import type { Worker, SystemInfo, Sweep, SpawnRequest } from './types'

const BASE = ''

/** Fetch with a timeout — prevents hung requests from blocking polling. */
async function fetchWithTimeout(url: string, opts?: RequestInit, timeoutMs = 5000): Promise<Response> {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...opts, signal: controller.signal })
  } finally {
    clearTimeout(id)
  }
}

export async function getWorkers(): Promise<Worker[]> {
  const res = await fetchWithTimeout(`${BASE}/api/workers`)
  return res.json()
}

export async function killWorker(pid: number): Promise<void> {
  await fetchWithTimeout(`${BASE}/api/workers/kill/${pid}`, { method: 'POST' })
}

export async function spawnWorkers(req: SpawnRequest): Promise<{ spawned: { pid: number; log_file: string }[] }> {
  const res = await fetchWithTimeout(`${BASE}/api/workers/spawn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  }, 30000) // spawn can take a while
  return res.json()
}

export async function getWorkerLogs(pid: number, lines = 100): Promise<string> {
  const res = await fetchWithTimeout(`${BASE}/api/workers/${pid}/logs?lines=${lines}`)
  const data = await res.json()
  return data.log
}

export async function getSweeps(): Promise<Sweep[]> {
  const res = await fetchWithTimeout(`${BASE}/api/sweeps`, undefined, 15000) // W&B API can be slow
  return res.json()
}

export async function getSystem(): Promise<SystemInfo> {
  const res = await fetchWithTimeout(`${BASE}/api/system`)
  return res.json()
}
