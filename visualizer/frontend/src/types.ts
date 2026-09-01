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

export const ATTRIBUTE_QUESTIONS: Record<AskAttribute, string> = {
  category: 'What kind of product are you looking for?',
  material: 'Do you have a material preference?',
  color: 'Do you have a color preference?',
  size: 'Are there any sizing or fit requirements?',
  style: 'What style or fit do you prefer?',
  brand: 'Do you have a preferred brand?',
  budget: 'What budget range should I use?',
  feature: 'Which product features matter most to you?',
  use_case: 'What will you mainly use it for?',
  other: 'What other details matter most to you?',
}

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
export type EmbeddingProvider = 'gemini' | 'siliconflow'
export type AgentName = 'baseline' | 'v1' | 'terrace'
export type Intent = 'discovery' | 'browsing' | 'buying'
export type SessionMode = 'human_as_agent' | 'human_as_simulator' | 'agent_simulator'

export interface IntentPreference {
  id: string
  facet: string | null
  operator: string | null
  value: unknown
  semantic_text: string | null
  commitment: string
  evidence_text: string
}

export interface IntentSnapshot {
  goal: string | null
  preferences: IntentPreference[]
  dont_care_facets: string[]
  version: number
}

export interface QueryUnderstandingDiff {
  goal: { before: string | null; after: string | null; changed: boolean }
  preferences: { added: IntentPreference[]; removed: IntentPreference[] }
  dont_care: { added: string[]; removed: string[] }
  version: { before: number; after: number }
}

export interface QueryUnderstandingEvent {
  stage: 'query_understanding'
  status: 'started' | 'completed' | 'reused' | 'failed'
  turn: number
  elapsed_ms?: number
  diff?: QueryUnderstandingDiff
  intent?: IntentSnapshot
  operations?: Array<Record<string, unknown>>
  interpretation_summary?: string
  error?: { type: string; message: string }
}

export interface CompiledQuerySnapshot {
  intent_version: number
  q_lex: string
  q_sem: string
  search_ready: boolean
  hard_constraints: Array<Record<string, unknown>>
  ranking_preferences: Array<Record<string, unknown>>
  dont_care_facets: string[]
  directives: Record<string, unknown>
  requires_clarification: boolean
  clarification_reason: string | null
}

export interface QueryCompilerEvent {
  stage: 'query_compiler'
  status: 'started' | 'completed' | 'reused' | 'failed'
  turn: number
  elapsed_ms?: number
  compiled_query?: CompiledQuerySnapshot
  error?: { type: string; message: string }
}

export interface IntentTransparencyDiagnostics {
  status: 'healthy' | 'degraded' | 'unavailable'
  reason_codes: string[]
  semantic_factor_count: number
  hard_factor_count: number
  top_all_hard_compliance: number | null
  top_mean_hard_factor_compliance: number | null
  active_facets: string[]
  dont_care_facets: string[]
  open_facets: string[]
}

export interface IntentTransparencyEstimate {
  intent_version: number
  goal: string | null
  transparency: number | null
  change: number | null
  direction: 'initial' | 'narrower' | 'broader' | 'stable' | 'moved' | 'unavailable'
  remaining_intent_volume: number | null
  catalog_reference_volume: number
  goal_reference_volume: number | null
  diagnostics: IntentTransparencyDiagnostics
}

export interface IntentTransparencyEvent {
  stage: 'intent_transparency'
  status: 'started' | 'completed' | 'reused' | 'fallback'
  turn: number
  elapsed_ms?: number
  estimate?: IntentTransparencyEstimate | null
  applied_transparency?: number | null
  error?: { type: string; message: string } | null
}

export interface PipelineProductPreview {
  title: string
}

export interface RetrievalRoutePreview {
  route: string
  available: boolean
  hit_count: number
  top_hits: PipelineProductPreview[]
}

export interface RetrievalEvent {
  stage: 'retrieval'
  status: 'started' | 'completed' | 'reused' | 'failed'
  turn: number
  elapsed_ms?: number
  routes?: RetrievalRoutePreview[]
  eligible_count?: number
  fused_count?: number
  error?: { type: string; message: string } | null
}

export interface RankingEvent {
  stage: 'ranking'
  status: 'started' | 'completed' | 'reused' | 'failed'
  turn: number
  elapsed_ms?: number
  mode?: string
  candidate_count?: number
  selected_products?: PipelineProductPreview[]
  natural_language_reason?: string | null
  error?: { type: string; message: string } | null
}

export type AgentPipelineEvent =
  | QueryUnderstandingEvent
  | QueryCompilerEvent
  | IntentTransparencyEvent
  | RetrievalEvent
  | RankingEvent

export interface SessionStartOptions {
  mode: SessionMode
  sampleId: string
  dataset: string
  replyModel: ReplyModel
  embeddingProvider: EmbeddingProvider
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
  queried_attribute?: AskAttribute | null
  recommendations: ProductSummary[]
  hit_rank: number | null
  subscore: number | null
  intent_before: Intent
  intent_after: Intent
  intent_changed: boolean
  recommendation_scores: Record<string, number | null>
}

export interface SessionMetrics {
  current_intent: Intent
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
  embedding_provider: EmbeddingProvider
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
    intent_description: Record<string, string | string[]> | null
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
