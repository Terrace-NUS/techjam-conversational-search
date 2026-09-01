<script setup lang="ts">
import { Bot, CheckCircle2, UserRound } from '@lucide/vue'
import { motion } from 'motion-v'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import ProductDetailDialog from './ProductDetailDialog.vue'
import QueryUnderstandingCard from './QueryUnderstandingCard.vue'
import {
  ATTRIBUTE_QUESTIONS,
  type IntentTransparencyEvent,
  type QueryCompilerEvent,
  type QueryUnderstandingEvent,
  type RankingEvent,
  type RetrievalEvent,
  type SimulatorSession,
} from '@/types'

type Actor = 'agent' | 'simulator'

const props = defineProps<{
  session: SimulatorSession
  queryUnderstanding: Record<number, QueryUnderstandingEvent>
  queryCompiler: Record<number, QueryCompilerEvent>
  intentTransparency: Record<number, IntentTransparencyEvent>
  retrieval: Record<number, RetrievalEvent>
  ranking: Record<number, RankingEvent>
  pendingAgentTurn: number | null
}>()
const emit = defineEmits<{
  animationComplete: [payload: { actor: Actor; turn: number }]
}>()

const displayedUser = ref<Record<string, string>>({})
const displayedAgent = ref<Record<string, string>>({})
const revealedRecommendations = ref<Record<number, boolean>>({})
const collapsedThinking = ref<Record<number, boolean>>({})
const timeline = ref<HTMLElement | null>(null)
const timers = new Map<string, number>()
const completedAnimations = new Set<string>()
let hasSynced = false
let pinnedToBottom = true
let resizeObserver: ResizeObserver | null = null
let smoothScrollTimer: number | null = null

function reducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
}

function stop(key: string) {
  const timer = timers.get(key)
  if (timer !== undefined) window.clearTimeout(timer)
  timers.delete(key)
}

function showImmediately(target: typeof displayedUser, key: string, text: string) {
  stop(key)
  target.value = { ...target.value, [key]: text }
}

function typeText(
  target: typeof displayedUser,
  key: string,
  text: string,
  onComplete?: () => void,
) {
  if (reducedMotion() || !hasSynced) {
    showImmediately(target, key, text)
    if (onComplete) queueMicrotask(onComplete)
    return
  }

  const current = target.value[key] ?? ''
  if (current === text) {
    if (!timers.has(key) && onComplete) queueMicrotask(onComplete)
    return
  }
  stop(key)
  const prefix = text.startsWith(current) ? current : ''
  target.value = { ...target.value, [key]: prefix }
  const step = text.length > 600 ? 6 : text.length > 300 ? 3 : 1
  const tick = () => {
    const shown = target.value[key] ?? ''
    if (shown.length >= text.length) {
      stop(key)
      onComplete?.()
      return
    }
    target.value = { ...target.value, [key]: text.slice(0, shown.length + step) }
    timers.set(key, window.setTimeout(tick, 16))
  }
  timers.set(key, window.setTimeout(tick, 16))
}

function completeAnimation(actor: Actor, turn: number) {
  const key = `${actor}-${turn}`
  if (completedAnimations.has(key)) return
  completedAnimations.add(key)
  if (actor === 'agent') {
    revealedRecommendations.value = { ...revealedRecommendations.value, [turn]: true }
  }
  void nextTick(() => emit('animationComplete', { actor, turn }))
}

function syncDisplayedText(session: SimulatorSession) {
  session.turns.forEach((turn, index) => {
    const userKey = `turn-${index}-user`
    const agentKey = `turn-${index}-agent`
    const userText = turn.user_message
    const agentText = turn.agent_message.trim()
      || (turn.ask_attribute ? ATTRIBUTE_QUESTIONS[turn.ask_attribute] : '')
    const turnNumber = index + 1

    // A previous generated simulator message moves from current_user_message
    // into turn.user_message when an agent turn is submitted.
    const movedFromCurrent = session.mode !== 'human_as_simulator'
      && `current-${index + 1}` in displayedUser.value
    if (session.mode === 'human_as_simulator') showImmediately(displayedUser, userKey, userText)
    else if (movedFromCurrent) showImmediately(displayedUser, userKey, userText)
    else typeText(displayedUser, userKey, userText)

    if (session.mode === 'human_as_agent' || !hasSynced) {
      showImmediately(displayedAgent, agentKey, agentText)
      revealedRecommendations.value = { ...revealedRecommendations.value, [turnNumber]: true }
    } else if (props.pendingAgentTurn === turnNumber) {
      revealedRecommendations.value = { ...revealedRecommendations.value, [turnNumber]: false }
      if (agentText) {
        collapsedThinking.value = { ...collapsedThinking.value, [turnNumber]: true }
        typeText(displayedAgent, agentKey, agentText, () => completeAnimation('agent', turnNumber))
      } else {
        queueMicrotask(() => completeAnimation('agent', turnNumber))
      }
    } else {
      showImmediately(displayedAgent, agentKey, agentText)
      revealedRecommendations.value = { ...revealedRecommendations.value, [turnNumber]: true }
    }
  })

  if (session.current_user_message) {
    typeText(
      displayedUser,
      `current-${session.current_turn}`,
      session.current_user_message,
      () => completeAnimation('simulator', session.current_turn),
    )
  }
  hasSynced = true
}

function userText(index: number, fallback: string) {
  return displayedUser.value[`turn-${index}-user`] ?? fallback
}

function agentText(index: number, fallback: string) {
  return displayedAgent.value[`turn-${index}-agent`] ?? fallback
}

function currentUserText(turn: number, fallback: string) {
  return displayedUser.value[`current-${turn}`] ?? fallback
}

function recommendationsVisible(turn: number) {
  return revealedRecommendations.value[turn] ?? false
}

watch(
  [() => props.session, () => props.pendingAgentTurn],
  ([session]) => syncDisplayedText(session),
  { immediate: true, deep: true },
)

function updatePinnedState() {
  if (smoothScrollTimer !== null) return
  pinnedToBottom = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 48
}

function scrollToBottom() {
  if (!pinnedToBottom) return
  if (smoothScrollTimer !== null) window.clearTimeout(smoothScrollTimer)
  smoothScrollTimer = window.setTimeout(() => {
    smoothScrollTimer = null
    updatePinnedState()
  }, 400)
  window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: reducedMotion() ? 'auto' : 'smooth',
  })
}

function stopFollowingOnWheel(event: WheelEvent) {
  if (event.deltaY >= 0) return
  if (smoothScrollTimer !== null) window.clearTimeout(smoothScrollTimer)
  smoothScrollTimer = null
  pinnedToBottom = false
}

onMounted(() => {
  updatePinnedState()
  window.addEventListener('scroll', updatePinnedState, { passive: true })
  window.addEventListener('wheel', stopFollowingOnWheel, { passive: true })
  resizeObserver = new ResizeObserver(scrollToBottom)
  if (timeline.value) resizeObserver.observe(timeline.value)
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', updatePinnedState)
  window.removeEventListener('wheel', stopFollowingOnWheel)
  resizeObserver?.disconnect()
  if (smoothScrollTimer !== null) window.clearTimeout(smoothScrollTimer)
  timers.forEach((timer) => window.clearTimeout(timer))
  timers.clear()
})
</script>

<template>
  <div ref="timeline">
    <Card class="min-h-[32rem]">
      <CardHeader class="border-b">
      <div class="flex items-center justify-between gap-3">
        <CardTitle class="text-base">Conversation</CardTitle>
        <Badge v-if="pendingAgentTurn" variant="secondary">Agent thinking…</Badge>
        <Badge v-else-if="session.status === 'initializing'" variant="secondary">
          {{ session.mode === 'human_as_simulator' ? 'Starting Agent…' : session.mode === 'agent_simulator' ? 'Starting both…' : 'Simulator replying…' }}
        </Badge>
        <Badge v-else-if="session.status === 'waiting_for_agent'" variant="secondary">
          {{ session.mode === 'agent_simulator' ? 'Ready for next turn' : 'Your turn' }}
        </Badge>
        <Badge v-else-if="session.status === 'waiting_for_simulator'" variant="secondary">Your reply</Badge>
        <Badge v-else :variant="session.status === 'hit' ? 'default' : 'outline'">
          {{ session.status === 'hit' ? 'Target found' : 'Session complete' }}
        </Badge>
      </div>
      </CardHeader>
      <CardContent class="space-y-7 py-6">
      <div v-for="(turn, index) in session.turns" :key="index" class="space-y-4">
        <motion.div
          :class="session.mode === 'human_as_simulator' ? 'ml-8 flex flex-row-reverse gap-3' : 'flex gap-3'"
          :initial="{ opacity: 0, x: -14 }"
          :animate="{ opacity: 1, x: 0 }"
          :transition="{ duration: 0.24, ease: 'easeOut' }"
        >
          <div :class="['grid size-8 shrink-0 place-items-center rounded-full', session.mode === 'human_as_simulator' ? 'border bg-card' : 'bg-primary text-primary-foreground']">
            <UserRound class="size-4" />
          </div>
          <div :class="['min-w-0 flex-1', session.mode === 'human_as_simulator' && 'flex flex-col items-end']">
            <p class="mb-1 text-xs font-medium text-muted-foreground">
              {{ session.mode === 'human_as_simulator' ? 'You · Simulator' : 'Simulator' }} · Turn {{ index + 1 }}
            </p>
            <p :class="['max-w-[90%] rounded-xl px-4 py-3 text-sm leading-6', session.mode === 'human_as_simulator' ? 'rounded-tr-sm border bg-card text-left' : 'rounded-tl-sm bg-muted']">
              {{ userText(index, turn.user_message) }}
            </p>
            <p v-if="turn.user_message_original" class="mt-1.5 text-xs leading-5 text-muted-foreground">
              Original: {{ turn.user_message_original }}
            </p>
          </div>
        </motion.div>

        <motion.div
          :class="session.mode === 'human_as_simulator' ? 'flex gap-3' : 'ml-8 flex flex-row-reverse gap-3'"
          :initial="{ opacity: 0, x: 14 }"
          :animate="{ opacity: 1, x: 0 }"
          :transition="{ duration: 0.26, delay: 0.08, ease: 'easeOut' }"
        >
          <div :class="['grid size-8 shrink-0 place-items-center rounded-full', session.mode === 'human_as_simulator' ? 'bg-primary text-primary-foreground' : 'border bg-card']">
            <Bot class="size-4" />
          </div>
          <div :class="['flex min-w-0 flex-1 flex-col space-y-2', session.mode === 'human_as_simulator' ? 'items-start' : 'items-end']">
            <div :class="['flex flex-wrap items-center gap-2', session.mode !== 'human_as_simulator' && 'justify-end']">
              <p class="text-xs font-medium text-muted-foreground">
                {{ session.mode === 'human_as_agent' ? 'You · Agent' : `${session.agent === 'terrace' ? 'Aperture' : session.agent?.toUpperCase()} · Agent` }}
              </p>
              <Badge v-if="turn.hit_rank" class="bg-emerald-600">
                <CheckCircle2 class="size-3" /> rank {{ turn.hit_rank }}
              </Badge>
              <Badge v-if="session.debug && turn.subscore !== null" variant="outline">
                score {{ turn.subscore.toFixed(3) }}
              </Badge>
              <Badge v-if="session.debug && turn.intent_changed" class="bg-violet-600">
                {{ turn.intent_before }} → {{ turn.intent_after }}
              </Badge>
            </div>
            <QueryUnderstandingCard
              v-if="queryUnderstanding[index + 1]"
              :progress="queryUnderstanding[index + 1]"
              :compiler="queryCompiler[index + 1]"
              :transparency="intentTransparency[index + 1]"
              :retrieval="retrieval[index + 1]"
              :ranking="ranking[index + 1]"
              :collapse="collapsedThinking[index + 1]"
            />
            <div
              v-if="turn.agent_message.trim() || turn.ask_attribute"
              :class="['max-w-[90%] space-y-2 rounded-xl border bg-card px-4 py-3 text-left text-sm leading-6', session.mode === 'human_as_simulator' ? 'rounded-tl-sm' : 'rounded-tr-sm']"
            >
              <p>{{ agentText(index, turn.agent_message.trim() || (turn.ask_attribute && ATTRIBUTE_QUESTIONS[turn.ask_attribute]) || '') }}</p>
              <p v-if="session.debug && session.mode !== 'human_as_simulator'" class="font-mono text-xs text-muted-foreground">
                queried attribute: {{ turn.queried_attribute === undefined ? 'resolving…' : (turn.queried_attribute ?? 'null') }}
              </p>
            </div>
            <div
              v-if="turn.recommendations.length && recommendationsVisible(index + 1)"
              :class="['flex flex-wrap gap-1.5', session.mode !== 'human_as_simulator' && 'justify-end']"
            >
              <motion.div
                v-for="(product, productIndex) in turn.recommendations"
                :key="product.parent_asin"
                :initial="{ opacity: 0, y: 8, scale: 0.98 }"
                :animate="{ opacity: 1, y: 0, scale: 1 }"
                :transition="{ duration: 0.24, delay: productIndex * 0.045, ease: 'easeOut' }"
              >
                <ProductDetailDialog :product="product">
                  <Badge
                    as="button"
                    type="button"
                    variant="secondary"
                    class="max-w-72 cursor-pointer"
                    :title="product.title"
                  >
                    <img
                      v-if="product.thumb"
                      :src="product.thumb"
                      alt=""
                      class="size-5 shrink-0 rounded-sm bg-white object-contain"
                      loading="lazy"
                      referrerpolicy="no-referrer"
                    />
                    <span class="truncate">#{{ productIndex + 1 }} {{ product.title }}</span>
                    <span v-if="session.debug" class="shrink-0 font-mono opacity-70">
                      · {{ turn.recommendation_scores[product.parent_asin] === null || turn.recommendation_scores[product.parent_asin] === undefined
                        ? 'score —'
                        : `score ${turn.recommendation_scores[product.parent_asin]?.toFixed(3)}` }}
                    </span>
                  </Badge>
                </ProductDetailDialog>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div
        v-if="session.current_user_message"
        :key="session.current_turn"
        class="flex gap-3"
        :initial="{ opacity: 0, x: -14 }"
        :animate="{ opacity: 1, x: 0 }"
        :transition="{ duration: 0.24, ease: 'easeOut' }"
      >
        <div class="grid size-8 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
          <UserRound class="size-4" />
        </div>
        <div class="min-w-0 flex-1">
          <p class="mb-1 text-xs font-medium text-muted-foreground">
            Simulator · Turn {{ session.current_turn }}
          </p>
          <p class="rounded-xl rounded-tl-sm bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground">
            {{ currentUserText(session.current_turn, session.current_user_message) }}
          </p>
          <p
            v-if="session.current_user_message_original"
            class="mt-1.5 text-xs leading-5 text-muted-foreground"
          >
            Original: {{ session.current_user_message_original }}
          </p>
        </div>
      </motion.div>

      <motion.div
        v-if="session.status === 'initializing'"
        class="flex gap-3"
        :initial="{ opacity: 0, x: -14 }"
        :animate="{ opacity: 1, x: 0 }"
      >
        <div class="grid size-8 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
          <UserRound class="size-4" />
        </div>
        <div>
          <p class="mb-1 text-xs font-medium text-muted-foreground">
            {{ session.mode === 'human_as_simulator' ? 'Agent is starting' : session.mode === 'agent_simulator' ? 'Agent and Simulator' : 'Simulator · Turn 1' }}
          </p>
          <p class="rounded-xl rounded-tl-sm bg-muted px-4 py-3 text-sm text-muted-foreground">
            {{ session.mode === 'human_as_simulator' ? 'Preparing Agent…' : session.mode === 'agent_simulator' ? 'Preparing conversation…' : 'Generating reply…' }}
          </p>
        </div>
      </motion.div>
      </CardContent>
    </Card>
  </div>
</template>
