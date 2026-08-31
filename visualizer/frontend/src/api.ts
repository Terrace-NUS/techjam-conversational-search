import type {
  AgentTurnInput,
  AgentName,
  CatalogFilters,
  CatalogSearchInput,
  DatasetOption,
  ProductSummary,
  ReplyModel,
  SampleSummary,
  SimulatorSession,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function getDatasets(): Promise<DatasetOption[]> {
  return request('/api/datasets')
}

export function getSamples(dataset: string): Promise<SampleSummary[]> {
  return request(`/api/samples?dataset=${encodeURIComponent(dataset)}`)
}

export function createSession(
  sampleId: string,
  dataset: string,
  replyModel: ReplyModel,
  debug: boolean,
): Promise<SimulatorSession> {
  return request('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ sample_id: sampleId, dataset, reply_model: replyModel, debug }),
  })
}

export function initializeSession(sessionId: string): Promise<SimulatorSession> {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}/initialize`, { method: 'POST' })
}

export function createAutoSession(
  sampleId: string,
  dataset: string,
  agent: AgentName,
  replyModel: ReplyModel,
  debug: boolean,
): Promise<SimulatorSession> {
  return request('/api/auto-sessions', {
    method: 'POST',
    body: JSON.stringify({ sample_id: sampleId, dataset, agent, reply_model: replyModel, debug }),
  })
}

export function initializeAutoSession(sessionId: string): Promise<SimulatorSession> {
  return request(`/api/auto-sessions/${encodeURIComponent(sessionId)}/initialize`, {
    method: 'POST',
  })
}

export function stepAutoSession(sessionId: string): Promise<SimulatorSession> {
  return request(`/api/auto-sessions/${encodeURIComponent(sessionId)}/step`, { method: 'POST' })
}

export function createHumanSession(
  sampleId: string,
  dataset: string,
  agent: AgentName,
): Promise<SimulatorSession> {
  return request('/api/human-sessions', {
    method: 'POST',
    body: JSON.stringify({ sample_id: sampleId, dataset, agent }),
  })
}

export function initializeHumanSession(sessionId: string): Promise<SimulatorSession> {
  return request(`/api/human-sessions/${encodeURIComponent(sessionId)}/initialize`, {
    method: 'POST',
  })
}

export function submitHumanReply(
  sessionId: string,
  message: string,
): Promise<SimulatorSession> {
  return request(`/api/human-sessions/${encodeURIComponent(sessionId)}/reply`, {
    method: 'POST',
    body: JSON.stringify({ message }),
  })
}

export function rewriteMessage(message: string): Promise<{ message: string }> {
  return request('/api/rewrite', { method: 'POST', body: JSON.stringify({ message }) })
}

export function submitAgentTurn(
  sessionId: string,
  input: AgentTurnInput,
): Promise<SimulatorSession> {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}/turn`, {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function getCatalogFilters(): Promise<CatalogFilters> {
  return request('/api/catalog/filters')
}

export function getProduct(parentAsin: string): Promise<ProductSummary> {
  return request(`/api/catalog/${encodeURIComponent(parentAsin)}`)
}

export function searchCatalog(
  input: CatalogSearchInput,
  signal?: AbortSignal,
): Promise<ProductSummary[]> {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries({ limit: 40, ...input })) {
    if (value !== undefined && value !== '') params.set(key, String(value))
  }
  return request(`/api/catalog/search?${params}`, { signal })
}
