import type {
  AgentTurnInput,
  CatalogFilters,
  CatalogSearchInput,
  ProductSummary,
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

export function getSamples(): Promise<SampleSummary[]> {
  return request('/api/samples')
}

export function createSession(sampleId: string): Promise<SimulatorSession> {
  return request('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ sample_id: sampleId }),
  })
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
