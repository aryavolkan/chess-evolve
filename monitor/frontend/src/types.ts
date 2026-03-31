export interface Worker {
  pid: number
  sweep_id: string | null
  cpu_pct: number
  mem_pct: number
  status: "running" | "idle"
  uptime_sec: number
  last_log_line: string
  log_file: string | null
}

export interface SystemInfo {
  cpu_cores: number
  cpu_percent: number
  mem_total_gb: number
  mem_used_gb: number
  mem_percent: number
}

export interface Sweep {
  id: string
  state: string
  run_count: number
  running: number
  best_metric: number | null
  url: string
}

export interface SpawnRequest {
  sweep_id: string
  count: number
  rayon_threads: number
}
