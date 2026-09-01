<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { ArrowRight, BrainCircuit, Check, ChevronDown, ListChecks, LoaderCircle, Minus, Plus, ScanSearch, SlidersHorizontal } from '@lucide/vue'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { nonlinearScore } from '@/lib/utils'
import type {
  IntentPreference,
  IntentTransparencyEvent,
  QueryCompilerEvent,
  QueryUnderstandingEvent,
  RankingEvent,
  RetrievalEvent,
} from '@/types'

const props = withDefaults(defineProps<{
  progress: QueryUnderstandingEvent
  compiler?: QueryCompilerEvent
  transparency?: IntentTransparencyEvent
  retrieval?: RetrievalEvent
  ranking?: RankingEvent
  card?: boolean
  collapse?: boolean
}>(), { card: false, collapse: false })
const expanded = ref(true)
const thinkingElapsedMs = ref(0)
const thinkingStartedAt = performance.now()
let thinkingTimer: number | null = null

function updateThinkingTime() {
  thinkingElapsedMs.value = performance.now() - thinkingStartedAt
}

function stopThinkingTimer() {
  if (thinkingTimer !== null) window.clearInterval(thinkingTimer)
  thinkingTimer = null
}

if (!props.card) {
  updateThinkingTime()
  thinkingTimer = window.setInterval(updateThinkingTime, 100)
}

watch(() => props.collapse, (collapse) => {
  if (!collapse) return
  expanded.value = false
  updateThinkingTime()
  stopThinkingTimer()
}, { immediate: true })

onBeforeUnmount(stopThinkingTimer)

const changed = computed(() => {
  const diff = props.progress.diff
  return Boolean(
    diff?.goal.changed
      || diff?.preferences.added.length
      || diff?.preferences.removed.length
      || diff?.dont_care.added.length
      || diff?.dont_care.removed.length,
  )
})

const preferenceChanges = computed(() => {
  const added = [...(props.progress.diff?.preferences.added ?? [])]
  const removed = [...(props.progress.diff?.preferences.removed ?? [])]
  const replacedFacets = new Set(
    (props.progress.operations ?? [])
      .filter((operation) => operation.op === 'replace_facet' && typeof operation.facet === 'string')
      .map((operation) => operation.facet as string),
  )
  const modified: Array<{ before: IntentPreference; after: IntentPreference }> = []

  for (const facet of replacedFacets) {
    while (true) {
      const beforeIndex = removed.findIndex((preference) => preference.facet === facet)
      const afterIndex = added.findIndex((preference) => preference.facet === facet)
      if (beforeIndex < 0 || afterIndex < 0) break
      modified.push({
        before: removed.splice(beforeIndex, 1)[0],
        after: added.splice(afterIndex, 1)[0],
      })
    }
  }

  return { added, removed, modified }
})

function preferenceValue(preference: IntentPreference): string {
  if (preference.facet === 'system_product_category') {
    return preference.semantic_text ?? preference.evidence_text ?? String(preference.value ?? '')
  }
  const value = preference.value ?? preference.semantic_text ?? preference.evidence_text
  return Array.isArray(value) ? value.join(', ') : String(value ?? '')
}

function facetLabel(facet: string | null): string {
  if (!facet) return 'Requirement'
  if (facet === 'system_product_category') return 'Category'
  return facet.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function operatorLabel(operator: string | null): string {
  return {
    eq: 'is',
    neq: 'is not',
    in: 'is one of',
    not_in: 'is none of',
    lt: 'is under',
    le: 'is at most',
    gt: 'is over',
    ge: 'is at least',
  }[operator ?? ''] ?? operator?.replaceAll('_', ' ') ?? ''
}

function preferenceLabel(preference: IntentPreference): string {
  return [facetLabel(preference.facet), operatorLabel(preference.operator), preferenceValue(preference)]
    .filter(Boolean)
    .join(' ')
}

function percent(value: number | null | undefined): string {
  return value == null ? 'Unavailable' : `${Math.round(value * 100)}%`
}

function focusPercent(value: number | null | undefined): string {
  return value == null ? 'Unavailable' : `${Math.round(nonlinearScore(value) * 100)}%`
}

function compactNumber(value: number | null | undefined): string {
  return value == null ? 'unavailable' : new Intl.NumberFormat('en', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value)
}

function statusLabel(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function routeLabel(value: string): string {
  return {
    dense: 'Semantic',
    lexical: 'Keyword',
    facet: 'Structured',
  }[value] ?? statusLabel(value)
}

function rankingModeLabel(value: string): string {
  return {
    deepseek_quality_dpp: 'Evidence ranking + diversity selection',
    deepseek_bge_fallback_dpp: 'Semantic ranking + diversity selection',
    bge_dpp: 'Semantic reranking + diversity selection',
    bge_dpp_after_failure: 'Semantic fallback + diversity selection',
    formal_mmr: 'Relevance + diversity selection',
    formal_mmr_fallback: 'Relevance + diversity fallback',
  }[value] ?? statusLabel(value)
}
</script>

<template>
  <div :class="[card ? 'overflow-hidden rounded-xl border bg-card shadow-sm' : 'w-full max-w-[90%]', 'text-left text-sm']">
    <button
      type="button"
      :class="['flex w-full items-center gap-2 text-muted-foreground', card ? 'px-6 py-5' : 'justify-end py-1']"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      <BrainCircuit class="size-4" />
      <span class="font-medium text-foreground">
        {{ card ? 'Query Understanding' : (collapse ? 'Thought' : 'Thinking') }}
      </span>
      <span v-if="!card" class="text-xs tabular-nums">
        for {{ (thinkingElapsedMs / 1000).toFixed(1) }}s
      </span>
      <span v-if="card" class="ml-auto text-xs">Turn {{ progress.turn }}</span>
      <ChevronDown :class="['size-4 transition-transform duration-200', expanded && 'rotate-180']" />
    </button>

    <div :class="['grid transition-[grid-template-rows,opacity] duration-300 ease-out', expanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0']">
      <div class="overflow-hidden">
        <div :class="[card ? 'border-t px-6 py-5' : 'mr-2 mt-2 border-r pr-5 pb-1', 'space-y-3']">
          <template v-if="card">
            <template v-if="progress.intent">
              <Transition name="qu-row" mode="out-in" appear>
                <div v-if="progress.intent.goal" :key="progress.intent.goal" class="space-y-1">
                  <p class="text-xs font-medium uppercase text-muted-foreground">Shopping goal</p>
                  <p class="text-base font-medium text-foreground">{{ progress.intent.goal }}</p>
                </div>
              </Transition>
              <div class="space-y-2">
                <p class="text-xs font-medium uppercase text-muted-foreground">Persistent preferences</p>
                <TransitionGroup name="qu-row" tag="div" class="relative space-y-1.5" appear>
                  <div
                    v-for="preference in progress.intent.preferences"
                    :key="`${preference.id}:${preferenceLabel(preference)}`"
                    class="grid gap-0.5 border-b border-border/60 py-2 last:border-0"
                  >
                    <span class="text-xs font-medium text-muted-foreground">{{ facetLabel(preference.facet) }}</span>
                    <span class="min-w-0 break-words text-foreground/90">
                      <span v-if="preference.operator" class="text-muted-foreground">{{ operatorLabel(preference.operator) }} </span>
                      {{ preferenceValue(preference) }}
                    </span>
                  </div>
                </TransitionGroup>
                <Transition name="qu-row" appear>
                  <p v-if="!progress.intent.preferences.length" class="text-muted-foreground">
                    No persistent preferences yet.
                  </p>
                </Transition>
              </div>
              <TransitionGroup
                v-if="progress.intent.dont_care_facets.length"
                name="qu-row"
                tag="div"
                class="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground"
                appear
              >
                <span v-for="facet in progress.intent.dont_care_facets" :key="facet">
                  Don't care: {{ facetLabel(facet) }}
                </span>
              </TransitionGroup>
            </template>
            <p v-else class="text-muted-foreground">Waiting for the first structured intent…</p>
          </template>

          <Transition v-else name="qu-content" mode="out-in" appear>
            <div :key="progress.status" class="flex items-start gap-3">
              <div class="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border bg-background">
                <LoaderCircle v-if="progress.status === 'started'" class="size-3 animate-spin" />
                <Check v-else-if="progress.status !== 'failed'" class="size-3 text-emerald-600" />
                <span v-else class="text-xs text-red-600">!</span>
              </div>
              <div class="min-w-0 flex-1 space-y-3">
              <div>
                <p class="font-medium text-foreground">Query understanding</p>
                <p v-if="progress.status === 'started'" class="mt-0.5 text-muted-foreground">
                  Structuring the latest request into intent and constraints…
                </p>
              </div>

              <p v-if="progress.status === 'failed'" class="text-red-600 dark:text-red-400">
                {{ progress.error?.message ?? 'Query understanding failed.' }}
              </p>

              <template v-else-if="progress.diff && progress.intent">
                <p v-if="progress.interpretation_summary" class="text-muted-foreground">
                  {{ progress.interpretation_summary }}
                </p>
                <div class="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                  <span>Intent v{{ progress.diff.version.before }} → v{{ progress.diff.version.after }}</span>
                  <span>{{ progress.status }}</span>
                </div>

                <Transition name="qu-row" appear>
                  <div v-if="progress.diff.goal.changed" class="space-y-1.5">
                    <p class="text-xs font-medium uppercase text-muted-foreground">Goal</p>
                    <div class="flex items-center gap-2">
                      <span class="text-red-600 line-through dark:text-red-400">{{ progress.diff.goal.before ?? 'unset' }}</span>
                      <ArrowRight class="size-3.5 shrink-0 text-muted-foreground" />
                      <span class="font-medium text-emerald-700 dark:text-emerald-400">{{ progress.diff.goal.after ?? 'unset' }}</span>
                    </div>
                  </div>
                </Transition>

                <TransitionGroup name="qu-row" tag="div" class="relative space-y-2" appear>
                  <div
                    v-for="change in preferenceChanges.modified"
                    :key="`modify:${change.before.id}:${change.after.id}`"
                    class="space-y-1.5"
                  >
                    <p class="text-xs font-medium uppercase text-muted-foreground">
                      Modified {{ facetLabel(change.after.facet) }}
                    </p>
                    <div class="flex items-start gap-2">
                      <span class="break-words text-red-600 line-through dark:text-red-400">{{ preferenceValue(change.before) }}</span>
                      <ArrowRight class="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                      <span class="break-words font-medium text-emerald-700 dark:text-emerald-400">{{ preferenceValue(change.after) }}</span>
                    </div>
                  </div>
                  <div
                    v-for="preference in preferenceChanges.added"
                    :key="`add:${preference.id}:${preferenceLabel(preference)}`"
                    class="flex items-start gap-2 text-emerald-700 dark:text-emerald-400"
                  >
                    <Plus class="mt-0.5 size-3.5 shrink-0" />
                    <span class="break-words">{{ preferenceLabel(preference) }}</span>
                  </div>
                  <div
                    v-for="preference in preferenceChanges.removed"
                    :key="`remove:${preference.id}:${preferenceLabel(preference)}`"
                    class="flex items-start gap-2 text-red-600 dark:text-red-400"
                  >
                    <Minus class="mt-0.5 size-3.5 shrink-0" />
                    <span class="break-words line-through">{{ preferenceLabel(preference) }}</span>
                  </div>
                </TransitionGroup>

                <TransitionGroup name="qu-row" tag="div" class="relative flex flex-wrap gap-x-4 gap-y-2" appear>
                  <span v-for="facet in progress.diff.dont_care.added" :key="`dc-add:${facet}`" class="text-emerald-700 dark:text-emerald-400">
                    + don't care {{ facetLabel(facet) }}
                  </span>
                  <span v-for="facet in progress.diff.dont_care.removed" :key="`dc-remove:${facet}`" class="text-red-600 dark:text-red-400">
                    - don't care {{ facetLabel(facet) }}
                  </span>
                </TransitionGroup>

                <p v-if="!changed" class="text-muted-foreground">No structured intent changes.</p>
                <Separator />
                <p class="text-xs text-muted-foreground">Persistent intent updated in the side panel.</p>
              </template>

              <Separator v-if="progress.status !== 'started'" class="-ml-8 w-[calc(100%+2rem)]" />
              <div v-if="progress.status !== 'started'" class="-ml-8 flex w-[calc(100%+2rem)] items-start gap-3">
                <div class="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border bg-background">
                  <LoaderCircle v-if="compiler?.status === 'started'" class="size-3 animate-spin" />
                  <Check v-else-if="compiler && compiler.status !== 'failed'" class="size-3 text-emerald-600" />
                  <span v-else-if="compiler?.status === 'failed'" class="text-xs text-red-600">!</span>
                  <SlidersHorizontal v-else class="size-3 text-muted-foreground" />
                </div>
                <div class="min-w-0 flex-1 space-y-2">
                  <div class="flex items-center justify-between gap-3">
                    <p class="font-medium text-foreground">Query compiler</p>
                    <span v-if="compiler?.elapsed_ms !== undefined" class="text-xs text-muted-foreground">
                      {{ compiler.elapsed_ms.toFixed(1) }}ms
                    </span>
                  </div>
                  <p v-if="!compiler || compiler.status === 'started'" class="text-muted-foreground">
                    {{ compiler ? 'Compiling retrieval instructions…' : 'Waiting for query understanding…' }}
                  </p>
                  <p v-else-if="compiler.status === 'failed'" class="text-red-600 dark:text-red-400">
                    {{ compiler.error?.message ?? 'Query compilation failed.' }}
                  </p>
                  <template v-else-if="compiler.compiled_query">
                    <div v-if="compiler.compiled_query.q_sem" class="space-y-0.5">
                      <p class="text-xs font-medium text-muted-foreground">Semantic query</p>
                      <p class="break-words text-foreground/90">{{ compiler.compiled_query.q_sem }}</p>
                    </div>
                    <div v-if="compiler.compiled_query.q_lex" class="space-y-0.5">
                      <p class="text-xs font-medium text-muted-foreground">Keyword query</p>
                      <p class="break-words text-foreground/90">{{ compiler.compiled_query.q_lex }}</p>
                    </div>
                    <p class="text-xs text-muted-foreground">
                      {{ compiler.compiled_query.hard_constraints.length }} hard filters ·
                      {{ compiler.compiled_query.ranking_preferences.length }} ranking signals
                    </p>
                  </template>
                </div>
              </div>

              <Separator
                v-if="compiler && compiler.status !== 'started' && compiler.status !== 'failed'"
                class="-ml-8 w-[calc(100%+2rem)]"
              />
              <div
                v-if="compiler && compiler.status !== 'started' && compiler.status !== 'failed'"
                class="-ml-8 flex w-[calc(100%+2rem)] items-start gap-3"
              >
                <div class="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border bg-background">
                  <LoaderCircle v-if="transparency?.status === 'started'" class="size-3 animate-spin" />
                  <Check v-else-if="transparency && transparency.status !== 'fallback'" class="size-3 text-emerald-600" />
                  <span v-else-if="transparency?.status === 'fallback'" class="text-xs text-amber-600">!</span>
                  <ScanSearch v-else class="size-3 text-muted-foreground" />
                </div>
                <div class="min-w-0 flex-1 space-y-2">
                  <div class="flex items-center justify-between gap-3">
                    <p class="font-medium text-foreground">Intent transparency</p>
                    <span v-if="transparency?.elapsed_ms !== undefined" class="text-xs text-muted-foreground">
                      {{ transparency.elapsed_ms.toFixed(1) }}ms
                    </span>
                  </div>
                  <p v-if="!transparency || transparency.status === 'started'" class="text-muted-foreground">
                    {{ transparency ? 'Scanning catalog intent volume…' : 'Waiting for compiled query…' }}
                  </p>
                  <template v-else-if="transparency.estimate">
                    <p class="text-xs leading-5 text-muted-foreground">
                      Catalog mass {{ compactNumber(transparency.estimate.catalog_reference_volume) }}
                      <ArrowRight class="mx-1 inline size-3" />
                      {{ transparency.estimate.diagnostics.semantic_factor_count }} semantic factors
                      <ArrowRight class="mx-1 inline size-3" />
                      {{ transparency.estimate.diagnostics.hard_factor_count }} hard filters
                      <ArrowRight class="mx-1 inline size-3" />
                      {{ compactNumber(transparency.estimate.remaining_intent_volume) }} remaining
                    </p>
                    <div class="flex flex-wrap gap-x-5 gap-y-1">
                      <p><span class="text-muted-foreground">Intent focus</span> <span class="font-semibold text-foreground">{{ focusPercent(transparency.estimate.transparency) }}</span></p>
                      <p>
                        <span class="text-muted-foreground">Diagnostic health</span>
                        <span class="font-semibold text-foreground">{{ statusLabel(transparency.estimate.diagnostics.status) }}</span>
                        <span v-if="transparency.estimate.diagnostics.top_all_hard_compliance !== null" class="text-muted-foreground">
                          · hard compliance {{ percent(transparency.estimate.diagnostics.top_all_hard_compliance) }}
                        </span>
                      </p>
                    </div>
                    <p class="text-xs text-muted-foreground">
                      {{ statusLabel(transparency.estimate.direction) }} from the previous turn
                    </p>
                  </template>
                  <p v-else class="text-amber-700 dark:text-amber-400">
                    Measurement unavailable; retrieval uses {{ percent(transparency.applied_transparency) }}.
                  </p>
                </div>
              </div>

              <Separator
                v-if="transparency && transparency.status !== 'started'"
                class="-ml-8 w-[calc(100%+2rem)]"
              />
              <Transition v-if="transparency && transparency.status !== 'started'" name="qu-row" appear>
                <div class="-ml-8 flex w-[calc(100%+2rem)] items-start gap-3">
                  <div class="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border bg-background">
                    <LoaderCircle v-if="!retrieval || retrieval.status === 'started'" class="size-3 animate-spin" />
                    <Check v-else-if="retrieval.status !== 'failed'" class="size-3 text-emerald-600" />
                    <span v-else class="text-xs text-red-600">!</span>
                  </div>
                  <div class="min-w-0 flex-1 space-y-2">
                    <div class="flex items-center justify-between gap-3">
                      <p class="font-medium text-foreground">Multi-route retrieval</p>
                      <span v-if="retrieval?.elapsed_ms !== undefined" class="text-xs text-muted-foreground">
                        {{ retrieval.elapsed_ms.toFixed(1) }}ms
                      </span>
                    </div>
                    <p v-if="!retrieval || retrieval.status === 'started'" class="text-muted-foreground">
                      Running semantic, keyword, and structured search…
                    </p>
                    <p v-else-if="retrieval.status === 'failed'" class="text-red-600 dark:text-red-400">
                      {{ retrieval.error?.message ?? 'Retrieval failed.' }}
                    </p>
                    <template v-else>
                      <div class="flex flex-wrap gap-1.5">
                        <Badge
                          v-for="route in retrieval.routes ?? []"
                          :key="route.route"
                          variant="secondary"
                          class="gap-1.5"
                        >
                          <Check v-if="route.available" class="size-3 text-emerald-600" />
                          <span v-else class="size-1.5 rounded-full bg-muted-foreground/60" />
                          {{ routeLabel(route.route) }} · {{ route.hit_count }}
                        </Badge>
                      </div>
                      <div class="grid gap-2 sm:grid-cols-3">
                        <div
                          v-for="route in retrieval.routes ?? []"
                          :key="`results:${route.route}`"
                          class="min-w-0 rounded-md border border-border/60 px-2.5 py-2"
                        >
                          <p class="text-xs font-medium text-muted-foreground">{{ routeLabel(route.route) }} results</p>
                          <p
                            v-for="(product, productIndex) in route.top_hits"
                            :key="`${route.route}:${productIndex}`"
                            class="mt-1 line-clamp-2 text-xs leading-5 text-foreground/90"
                          >
                            {{ product.title }}
                          </p>
                          <p v-if="!route.top_hits.length" class="mt-1 text-xs text-muted-foreground">No matches</p>
                        </div>
                      </div>
                      <p class="text-xs text-muted-foreground">
                        {{ retrieval.eligible_count ?? 0 }} eligible products ·
                        {{ retrieval.fused_count ?? 0 }} candidates merged for ranking
                      </p>
                    </template>
                  </div>
                </div>
              </Transition>

              <Separator
                v-if="retrieval && retrieval.status !== 'started' && retrieval.status !== 'failed'"
                class="-ml-8 w-[calc(100%+2rem)]"
              />
              <Transition
                v-if="retrieval && retrieval.status !== 'started' && retrieval.status !== 'failed'"
                name="qu-row"
                appear
              >
                <div class="-ml-8 flex w-[calc(100%+2rem)] items-start gap-3">
                  <div class="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border bg-background">
                    <LoaderCircle v-if="!ranking || ranking.status === 'started'" class="size-3 animate-spin" />
                    <Check v-else-if="ranking.status !== 'failed'" class="size-3 text-emerald-600" />
                    <span v-else class="text-xs text-red-600">!</span>
                  </div>
                  <div class="min-w-0 flex-1 space-y-2">
                    <div class="flex items-center justify-between gap-3">
                      <p class="font-medium text-foreground">Ranking &amp; selection</p>
                      <span v-if="ranking?.elapsed_ms !== undefined" class="text-xs text-muted-foreground">
                        {{ ranking.elapsed_ms.toFixed(1) }}ms
                      </span>
                    </div>
                    <p v-if="!ranking || ranking.status === 'started'" class="text-muted-foreground">
                      Comparing candidate fit and diversity…
                    </p>
                    <p v-else-if="ranking.status === 'failed'" class="text-red-600 dark:text-red-400">
                      {{ ranking.error?.message ?? 'Ranking failed.' }}
                    </p>
                    <template v-else>
                      <p class="text-xs text-muted-foreground">
                        {{ rankingModeLabel(ranking.mode ?? 'ranking') }} ·
                        {{ ranking.candidate_count ?? 0 }} candidates →
                        {{ ranking.selected_products?.length ?? 0 }} selected
                      </p>
                      <div v-if="ranking.selected_products?.length" class="space-y-1">
                        <p
                          v-for="(product, productIndex) in ranking.selected_products.slice(0, 4)"
                          :key="`selected:${productIndex}`"
                          class="flex min-w-0 items-start gap-2 text-xs leading-5 text-foreground/90"
                        >
                          <ListChecks class="mt-1 size-3 shrink-0 text-emerald-600" />
                          <span class="line-clamp-2">{{ product.title }}</span>
                        </p>
                        <p v-if="ranking.selected_products.length > 4" class="pl-5 text-xs text-muted-foreground">
                          +{{ ranking.selected_products.length - 4 }} more selected
                        </p>
                      </div>
                      <p v-if="ranking.natural_language_reason" class="border-l-2 border-emerald-500/50 pl-3 text-xs leading-5 text-muted-foreground">
                        {{ ranking.natural_language_reason }}
                      </p>
                    </template>
                  </div>
                </div>
              </Transition>
              </div>
            </div>
          </Transition>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.qu-row-enter-active,
.qu-row-leave-active,
.qu-row-move {
  transition: opacity 240ms ease, transform 240ms ease;
}

.qu-row-enter-from,
.qu-row-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.qu-row-leave-active {
  position: absolute;
}

.qu-content-enter-active,
.qu-content-leave-active {
  transition: opacity 220ms ease, transform 220ms ease;
}

.qu-content-enter-from {
  opacity: 0;
  transform: translateY(6px);
}

.qu-content-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
