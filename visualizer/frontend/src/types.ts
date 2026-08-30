export const ASK_ATTRIBUTES = [
  'category',
  'material',
  'color',
  'size',
  'style',
  'brand',
  'budget',
  'feature',
  'use_case',
  'other',
] as const

export type AskAttribute = (typeof ASK_ATTRIBUTES)[number]

export interface SampleSummary {
  sample_id: string
  scenario_type: string
  difficulty_bucket: string
  category_bucket: string
}

export interface ProductSummary {
  parent_asin: string
  title: string
  thumb: string | null
  features?: string[] | null
  description?: string[] | null
  details?: Record<string, string> | null
  price: number | string | null
  categories: string[]
  store: string | null
  average_rating: number | null
  rating_number: number | null
}

export interface NumericRange {
  min: number | null
  max: number | null
}

export interface CatalogFilters {
  categories: string[]
  stores: string[]
  price: NumericRange
  average_rating: NumericRange
  rating_number: NumericRange
}

export interface CatalogSearchInput {
  q: string
  category?: string
  store?: string
  min_price?: number
  max_price?: number
  min_rating?: number
  min_rating_count?: number
  limit?: number
  offset?: number
}

export interface UserProfile {
  purchase_frequency: string
  average_prior_rating: number | null
  rating_style: string
  preference_tags: string[]
  summary: string
}

export interface TurnRecord {
  user_message: string
  agent_message: string
  ask_attribute: AskAttribute | null
  recommendations: ProductSummary[]
  hit_rank: number | null
}

export interface SessionOutcome {
  hit: boolean
  first_hit_turn: number | null
  best_rank: number | null
  reciprocal_rank: number
  target_product: ProductSummary
}

export interface SimulatorSession {
  id: string
  status: 'waiting_for_agent' | 'hit' | 'exhausted'
  sample: SampleSummary
  user_profile: UserProfile
  current_turn: number
  current_user_message: string | null
  turns: TurnRecord[]
  outcome: SessionOutcome | null
}

export interface AgentTurnInput {
  message: string
  ask_attribute: AskAttribute | null
  recommendations: string[]
}
