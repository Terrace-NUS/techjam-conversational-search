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

export interface DatasetOption {
  id: string
  label: string
  sample_count: number
  default: boolean
}

export type ReplyModel = 'template' | 'deepseek'
export type AgentName = 'baseline' | 'v1'
export type SessionMode = 'human_as_agent' | 'human_as_simulator' | 'agent_simulator'

export interface SessionStartOptions {
  mode: SessionMode
  sampleId: string
  dataset: string
  replyModel: ReplyModel
  agent: AgentName
  debug: boolean
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
  user_message_original: string | null
  agent_message: string
  ask_attribute: AskAttribute | null
  recommendations: ProductSummary[]
  hit_rank: number | null
  subscore: number | null
  intent_before: 'browsing' | 'buying'
  intent_after: 'browsing' | 'buying'
  intent_changed: boolean
  recommendation_scores: Record<string, number | null>
}

export interface SessionMetrics {
  current_intent: 'browsing' | 'buying'
  threshold: number
  last_subscore: number | null
  score_error: string | null
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
  mode: SessionMode
  status:
    | 'initializing'
    | 'waiting_for_agent'
    | 'waiting_for_simulator'
    | 'hit'
    | 'exhausted'
    | 'error'
  dataset: string
  reply_model: ReplyModel | null
  agent?: AgentName
  debug: boolean
  initialization_error: string | null
  debug_target_product: ProductSummary | null
  sample: SampleSummary
  user_profile: UserProfile
  current_turn: number
  current_user_message: string | null
  current_user_message_original: string | null
  metrics: SessionMetrics
  human_context?: {
    intent: string
    override: boolean
    intent_description: Record<string, string> | null
    fake_attributes: Record<string, unknown>
    correction_messages: Record<string, unknown>
    modify_turn: number | null
  }
  turns: TurnRecord[]
  outcome: SessionOutcome | null
}

export interface AgentTurnInput {
  message: string
  ask_attribute: AskAttribute | null
  recommendations: string[]
}
