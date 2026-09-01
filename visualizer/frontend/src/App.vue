<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { CircleAlert, Moon, Palette, Sparkles, Sun } from '@lucide/vue'
import { AnimatePresence, motion, MotionConfig, useAnimate } from 'motion-v'
import {
  createAutoSession,
  createHumanSession,
  createSession,
  getDatasets,
  getSamples,
  initializeAutoSession,
  initializeHumanSession,
  initializeSession,
  stepAutoSession,
  submitAgentTurn,
  submitHumanReply,
  watchSessionEvents,
} from './api'
import AgentComposer from './components/AgentComposer.vue'
import AutoConversationControls from './components/AutoConversationControls.vue'
import ConversationTimeline from './components/ConversationTimeline.vue'
import HumanSimulatorComposer from './components/HumanSimulatorComposer.vue'
import QueryUnderstandingCard from './components/QueryUnderstandingCard.vue'
import SessionContext from './components/SessionContext.vue'
import SessionSetup from './components/SessionSetup.vue'
import TransparencyCard from './components/TransparencyCard.vue'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type {
  AgentTurnInput,
  DatasetOption,
  ProductSummary,
  IntentTransparencyEvent,
  QueryCompilerEvent,
  QueryUnderstandingEvent,
  RankingEvent,
  RetrievalEvent,
  SampleSummary,
  SessionStartOptions,
  SimulatorSession,
} from './types'
import { ATTRIBUTE_QUESTIONS } from './types'

type ThemePalette = 'default' | 'claude' | 'whatsapp' | 'discord'
type Actor = 'agent' | 'simulator'

const samples = ref<SampleSummary[]>([])
const datasets = ref<DatasetOption[]>([])
const session = ref<SimulatorSession | null>(null)
const loading = ref(false)
const autoRunning = ref(false)
const autoActor = ref<Actor>('agent')
const pendingAutoSession = ref<SimulatorSession | null>(null)
const pendingAgentTurn = ref<number | null>(null)
const queryUnderstanding = ref<Record<number, QueryUnderstandingEvent>>({})
const queryCompiler = ref<Record<number, QueryCompilerEvent>>({})
const intentTransparency = ref<Record<number, IntentTransparencyEvent>>({})
const retrieval = ref<Record<number, RetrievalEvent>>({})
const ranking = ref<Record<number, RankingEvent>>({})
const persistentQueryUnderstanding = ref<QueryUnderstandingEvent | null>(null)
const persistentTransparency = ref<IntentTransparencyEvent | null>(null)
const error = ref('')
const composer = ref<InstanceType<typeof AgentComposer> | null>(null)
const darkMode = ref(document.documentElement.classList.contains('dark'))
const themePalette = ref<ThemePalette>(
  document.documentElement.classList.contains('theme-discord')
    ? 'discord'
    : document.documentElement.classList.contains('theme-whatsapp')
      ? 'whatsapp'
      : document.documentElement.classList.contains('theme-claude') ? 'claude' : 'default',
)
const [appScope, animateApp] = useAnimate()
let sessionEvents: EventSource | null = null
const completedAnimations = new Set<string>()
const pipelinePlayback = new Map<number, Promise<void>>()
const pipelineWaiters = new Map<number, {
  promise: Promise<void>
  resolve: () => void
  timer: number
}>()
let animationWaiter: {
  actor: Actor
  turn: number
  resolve: () => void
  timer: number
} | null = null

function clearAnimationWaiter() {
  if (!animationWaiter) return
  window.clearTimeout(animationWaiter.timer)
  animationWaiter.resolve()
  animationWaiter = null
}

function waitForAnimation(actor: Actor, turn: number): Promise<void> {
  clearAnimationWaiter()
  const key = `${actor}-${turn}`
  if (completedAnimations.delete(key)) return Promise.resolve()
  return new Promise((resolve) => {
    const finish = () => {
      if (animationWaiter?.actor === actor && animationWaiter.turn === turn) {
        animationWaiter = null
      }
      resolve()
    }
    animationWaiter = {
      actor,
      turn,
      resolve: finish,
      timer: window.setTimeout(finish, 30_000),
    }
  })
}

function handleAnimationComplete(payload: { actor: Actor; turn: number }) {
  const key = `${payload.actor}-${payload.turn}`
  if (animationWaiter?.actor !== payload.actor || animationWaiter.turn !== payload.turn) {
    completedAnimations.add(key)
    return
  }
  completedAnimations.delete(key)
  window.clearTimeout(animationWaiter.timer)
  animationWaiter.resolve()
}

function beginPipelineWait(turn: number) {
  const existing = pipelineWaiters.get(turn)
  if (existing) {
    window.clearTimeout(existing.timer)
    existing.resolve()
  }
  let resolve = () => {}
  const promise = new Promise<void>((done) => { resolve = done })
  const timer = window.setTimeout(() => finishPipelineWait(turn), 30_000)
  pipelineWaiters.set(turn, { promise, resolve, timer })
}

function finishPipelineWait(turn: number) {
  const waiter = pipelineWaiters.get(turn)
  if (!waiter) return
  window.clearTimeout(waiter.timer)
  pipelineWaiters.delete(turn)
  waiter.resolve()
}

function waitForPipeline(turn: number): Promise<void> {
  return pipelineWaiters.get(turn)?.promise ?? Promise.resolve()
}

function clearPipelinePlayback() {
  pipelinePlayback.clear()
  pipelineWaiters.forEach((waiter) => {
    window.clearTimeout(waiter.timer)
    waiter.resolve()
  })
  pipelineWaiters.clear()
}

function beginAgentTurn(current: SimulatorSession, turn: number) {
  pendingAgentTurn.value = turn
  if (current.agent === 'terrace') {
    beginPipelineWait(turn)
    queryUnderstanding.value = {
      ...queryUnderstanding.value,
      [turn]: { stage: 'query_understanding', status: 'started', turn },
    }
  }
}

function stageAgentTurn(
  current: SimulatorSession,
  userMessage: string,
  userMessageOriginal: string | null,
) {
  session.value = {
    ...current,
    current_user_message: null,
    current_user_message_original: null,
    turns: [
      ...current.turns,
      {
        user_message: userMessage,
        user_message_original: userMessageOriginal,
        agent_message: '',
        ask_attribute: null,
        recommendations: [],
        hit_rank: null,
        subscore: null,
        intent_before: current.metrics.current_intent,
        intent_after: current.metrics.current_intent,
        intent_changed: false,
        recommendation_scores: {},
      },
    ],
  }
}

function delay(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function queueCompilerEvent(current: SimulatorSession, event: QueryCompilerEvent) {
  const previous = pipelinePlayback.get(event.turn) ?? Promise.resolve()
  const playback = previous.then(async () => {
    if (session.value?.id !== current.id) return
    if (event.status !== 'started') {
      queryCompiler.value = {
        ...queryCompiler.value,
        [event.turn]: { stage: 'query_compiler', status: 'started', turn: event.turn },
      }
      await delay(2000)
      if (session.value?.id !== current.id) return
    }
    queryCompiler.value = { ...queryCompiler.value, [event.turn]: event }
    if (event.status === 'failed') finishPipelineWait(event.turn)
  })
  pipelinePlayback.set(event.turn, playback)
}

function queueTransparencyEvent(current: SimulatorSession, event: IntentTransparencyEvent) {
  const previous = pipelinePlayback.get(event.turn) ?? Promise.resolve()
  const playback = previous.then(async () => {
    if (session.value?.id !== current.id) return
    if (event.status !== 'started') {
      intentTransparency.value = {
        ...intentTransparency.value,
        [event.turn]: { stage: 'intent_transparency', status: 'started', turn: event.turn },
      }
      await delay(2000)
      if (session.value?.id !== current.id) return
    }
    intentTransparency.value = { ...intentTransparency.value, [event.turn]: event }
    if (event.status === 'completed' || event.status === 'reused') {
      persistentTransparency.value = event
    }
  })
  pipelinePlayback.set(event.turn, playback)
}

function queueRetrievalEvent(current: SimulatorSession, event: RetrievalEvent) {
  const previous = pipelinePlayback.get(event.turn) ?? Promise.resolve()
  const playback = previous.then(async () => {
    if (session.value?.id !== current.id) return
    if (event.status !== 'started') {
      retrieval.value = {
        ...retrieval.value,
        [event.turn]: { stage: 'retrieval', status: 'started', turn: event.turn },
      }
      await delay(2000)
      if (session.value?.id !== current.id) return
    }
    retrieval.value = { ...retrieval.value, [event.turn]: event }
    if (event.status === 'failed') finishPipelineWait(event.turn)
  })
  pipelinePlayback.set(event.turn, playback)
}

function queueRankingEvent(current: SimulatorSession, event: RankingEvent) {
  const previous = pipelinePlayback.get(event.turn) ?? Promise.resolve()
  const playback = previous.then(async () => {
    if (session.value?.id !== current.id) return
    if (event.status !== 'started') {
      ranking.value = {
        ...ranking.value,
        [event.turn]: { stage: 'ranking', status: 'started', turn: event.turn },
      }
      await delay(2000)
      if (session.value?.id !== current.id) return
    }
    ranking.value = { ...ranking.value, [event.turn]: event }
    if (event.status !== 'started') finishPipelineWait(event.turn)
  })
  pipelinePlayback.set(event.turn, playback)
}

function connectSessionEvents(current: SimulatorSession) {
  sessionEvents?.close()
  sessionEvents = null
  queryUnderstanding.value = {}
  queryCompiler.value = {}
  intentTransparency.value = {}
  retrieval.value = {}
  ranking.value = {}
  persistentQueryUnderstanding.value = null
  persistentTransparency.value = null
  clearPipelinePlayback()
  if (current.agent !== 'terrace') return
  sessionEvents = watchSessionEvents(current.id, (event) => {
    if (session.value?.id !== current.id) return
    if (event.stage === 'query_understanding') {
      queryUnderstanding.value = { ...queryUnderstanding.value, [event.turn]: event }
      if (event.status === 'completed' || event.status === 'reused') {
        persistentQueryUnderstanding.value = event
      }
      if (event.status === 'failed') finishPipelineWait(event.turn)
    } else if (event.stage === 'query_compiler') {
      queueCompilerEvent(current, event)
    } else if (event.stage === 'intent_transparency') {
      queueTransparencyEvent(current, event)
    } else if (event.stage === 'retrieval') {
      queueRetrievalEvent(current, event)
    } else {
      queueRankingEvent(current, event)
    }
  })
}

onBeforeUnmount(() => {
  sessionEvents?.close()
  clearAnimationWaiter()
  clearPipelinePlayback()
})

watch(themePalette, (palette) => {
  document.documentElement.classList.toggle('theme-claude', palette === 'claude')
  document.documentElement.classList.toggle('theme-whatsapp', palette === 'whatsapp')
  document.documentElement.classList.toggle('theme-discord', palette === 'discord')
  localStorage.setItem('theme-palette', palette)
})

function toggleTheme() {
  const nextDarkMode = !darkMode.value
  darkMode.value = nextDarkMode
  document.documentElement.classList.toggle('dark', nextDarkMode)
  localStorage.setItem('theme', nextDarkMode ? 'dark' : 'light')
  void animateApp(
    appScope.value,
    { opacity: [0.76, 1] },
    { duration: 0.28, ease: 'easeOut' },
  )
}

onMounted(async () => {
  loading.value = true
  try {
    datasets.value = await getDatasets()
    const dataset = datasets.value.find((item) => item.default)?.id ?? datasets.value[0]?.id
    samples.value = dataset ? await getSamples(dataset) : []
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load public cases'
  } finally {
    loading.value = false
  }
})

async function changeDataset(dataset: string) {
  loading.value = true
  error.value = ''
  try {
    samples.value = await getSamples(dataset)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load public cases'
  } finally {
    loading.value = false
  }
}

async function start(options: SessionStartOptions) {
  autoRunning.value = false
  loading.value = true
  error.value = ''
  try {
    const created = options.mode === 'human_as_agent'
      ? await createSession(options.sampleId, options.dataset, options.replyModel, options.embeddingProvider, options.debug)
      : options.mode === 'human_as_simulator'
        ? await createHumanSession(options.sampleId, options.dataset, options.agent, options.embeddingProvider)
        : await createAutoSession(
            options.sampleId,
            options.dataset,
            options.agent,
            options.replyModel,
            options.embeddingProvider,
            options.debug,
          )
    session.value = created
    connectSessionEvents(created)
    await nextTick()
    const initialized = options.mode === 'human_as_agent'
      ? await initializeSession(created.id)
      : options.mode === 'human_as_simulator'
        ? await initializeHumanSession(created.id)
        : await initializeAutoSession(created.id)
    if (session.value?.id !== created.id) return
    session.value = initialized
    autoActor.value = 'agent'
    pendingAutoSession.value = null
    pendingAgentTurn.value = null
    if (session.value?.status === 'error') {
      throw new Error(session.value.initialization_error ?? 'Could not generate first reply')
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not start session'
  } finally {
    loading.value = false
  }
}

async function advanceAuto(): Promise<boolean> {
  if (!session.value || session.value.mode !== 'agent_simulator') return false
  const sessionId = session.value.id
  error.value = ''

  if (autoActor.value === 'simulator') {
    const nextSession = pendingAutoSession.value
    if (!nextSession || nextSession.id !== sessionId) return false
    loading.value = true
    try {
      session.value = nextSession
      pendingAutoSession.value = null
      await nextTick()
      if (nextSession.current_user_message) {
        await waitForAnimation('simulator', nextSession.current_turn)
      }
      autoActor.value = 'agent'
      return nextSession.status === 'waiting_for_agent'
    } finally {
      loading.value = false
    }
  }

  const previousSession = session.value
  const turn = previousSession.current_turn
  const userMessage = previousSession.current_user_message ?? ''
  const userMessageOriginal = previousSession.current_user_message_original
  loading.value = true
  beginAgentTurn(previousSession, turn)
  stageAgentTurn(previousSession, userMessage, userMessageOriginal)
  try {
    const nextSession = await stepAutoSession(sessionId)
    if (session.value?.id !== sessionId) return false
    await waitForPipeline(turn)
    const newTurn = nextSession.turns.at(-1)
    if (!newTurn) throw new Error('Agent response did not include a turn')
    session.value = {
      ...nextSession,
      status: 'waiting_for_agent',
      current_turn: turn,
      current_user_message: null,
      current_user_message_original: null,
      turns: [...nextSession.turns.slice(0, -1), newTurn],
    }
    await nextTick()
    await waitForAnimation('agent', turn)
    pendingAgentTurn.value = null

    if (!nextSession.current_user_message) {
      session.value = nextSession
      return false
    }
    await delay(1000)
    pendingAutoSession.value = nextSession
    autoActor.value = 'simulator'
    return true
  } catch (cause) {
    session.value = previousSession
    pendingAgentTurn.value = null
    error.value = cause instanceof Error ? cause.message : 'Could not run the next turn'
    return false
  } finally {
    loading.value = false
  }
}

async function toggleAuto() {
  if (autoRunning.value) {
    autoRunning.value = false
    return
  }
  autoRunning.value = true
  try {
    while (autoRunning.value && await advanceAuto()) {}
  } finally {
    autoRunning.value = false
  }
}

async function replyAsSimulator(message: string) {
  if (!session.value) return
  const previousSession = session.value
  const turn = previousSession.current_turn
  loading.value = true
  error.value = ''
  beginAgentTurn(previousSession, turn)
  stageAgentTurn(previousSession, message, null)
  try {
    const nextSession = await submitHumanReply(previousSession.id, message)
    await waitForPipeline(turn)
    session.value = nextSession
    await nextTick()
    await waitForAnimation('agent', turn)
  } catch (cause) {
    session.value = previousSession
    error.value = cause instanceof Error ? cause.message : 'Could not get Agent response'
  } finally {
    pendingAgentTurn.value = null
    loading.value = false
  }
}

async function submit(input: AgentTurnInput, products: ProductSummary[]) {
  if (!session.value) return
  const previousSession = session.value
  loading.value = true
  error.value = ''
  try {
    const response = submitAgentTurn(previousSession.id, input)
    session.value = {
      ...previousSession,
      current_user_message: null,
      turns: [
        ...previousSession.turns,
        {
          user_message: previousSession.current_user_message ?? '',
          user_message_original: previousSession.current_user_message_original,
          agent_message: input.message || (input.ask_attribute ? ATTRIBUTE_QUESTIONS[input.ask_attribute] : ''),
          ask_attribute: input.ask_attribute,
          recommendations: products,
          hit_rank: null,
          subscore: null,
          intent_before: previousSession.metrics.current_intent,
          intent_after: previousSession.metrics.current_intent,
          intent_changed: false,
          recommendation_scores: {},
        },
      ],
    }
    await nextTick()
    const nextSession = await response
    session.value = nextSession
    composer.value?.clear()
    await nextTick()
    if (nextSession.current_user_message) {
      await waitForAnimation('simulator', nextSession.current_turn)
    }
  } catch (cause) {
    session.value = previousSession
    error.value = cause instanceof Error ? cause.message : 'Could not submit turn'
  } finally {
    loading.value = false
  }
}

function reset() {
  autoRunning.value = false
  autoActor.value = 'agent'
  pendingAutoSession.value = null
  pendingAgentTurn.value = null
  completedAnimations.clear()
  clearAnimationWaiter()
  sessionEvents?.close()
  sessionEvents = null
  queryUnderstanding.value = {}
  queryCompiler.value = {}
  intentTransparency.value = {}
  persistentQueryUnderstanding.value = null
  persistentTransparency.value = null
  clearPipelinePlayback()
  session.value = null
  error.value = ''
}
</script>

<template>
  <MotionConfig reduced-motion="user">
  <div ref="appScope" class="app-shell min-h-screen">
    <header class="border-b border-border/80 bg-background/80 backdrop-blur">
      <div class="mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-5 py-4 lg:px-8">
        <div class="flex items-center gap-3">
          <div class="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground shadow-sm">
            <Sparkles class="size-4" />
          </div>
          <div>
            <h1 class="font-semibold tracking-tight">APERTURE</h1>
            <p class="text-xs text-muted-foreground">Simulator visualizer</p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <Select v-model="themePalette">
            <SelectTrigger class="w-[8.5rem]" aria-label="Theme palette">
              <Palette class="size-4" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">Default</SelectItem>
              <SelectItem value="claude">Claude</SelectItem>
              <SelectItem value="whatsapp">Whatsapp</SelectItem>
              <SelectItem value="discord">Discord</SelectItem>
            </SelectContent>
          </Select>
          <Button
            size="icon-sm"
            variant="outline"
            :aria-label="darkMode ? 'Use light theme' : 'Use dark theme'"
            :title="darkMode ? 'Use light theme' : 'Use dark theme'"
            @click="toggleTheme"
          >
            <AnimatePresence mode="wait" :initial="false">
              <motion.span
                :key="darkMode ? 'sun' : 'moon'"
                class="flex"
                :initial="{ opacity: 0, rotate: -90, scale: 0.6 }"
                :animate="{ opacity: 1, rotate: 0, scale: 1 }"
                :exit="{ opacity: 0, rotate: 90, scale: 0.6 }"
                :transition="{ duration: 0.2, ease: 'easeOut' }"
              >
                <Sun v-if="darkMode" class="size-4" />
                <Moon v-else class="size-4" />
              </motion.span>
            </AnimatePresence>
          </Button>
        </div>
      </div>
    </header>

    <main class="mx-auto max-w-[1500px] px-5 py-8 lg:px-8">
      <div
        v-if="error"
        role="alert"
        class="mx-auto mb-5 flex max-w-2xl items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200"
      >
        <CircleAlert class="mt-0.5 size-4 shrink-0" />
        {{ error }}
      </div>

      <AnimatePresence mode="wait" :initial="false">
        <motion.div
          v-if="!session"
          key="session-setup"
          class="grid min-h-[calc(100vh-11rem)] place-items-center"
          :initial="{ opacity: 0, y: 12, scale: 0.99 }"
          :animate="{ opacity: 1, y: 0, scale: 1 }"
          :exit="{ opacity: 0, y: -8, scale: 0.99 }"
          :transition="{ duration: 0.24, ease: 'easeOut' }"
        >
          <SessionSetup
            :samples="samples"
            :datasets="datasets"
            :loading="loading"
            @dataset-change="changeDataset"
            @start="start"
          />
        </motion.div>

        <motion.div
          v-else
          :key="session.id"
          class="grid items-start gap-5 xl:grid-cols-[280px_minmax(0,1fr)_390px]"
          :initial="{ opacity: 0, y: 16 }"
          :animate="{ opacity: 1, y: 0 }"
          :exit="{ opacity: 0, y: 8 }"
          :transition="{ duration: 0.28, ease: 'easeOut' }"
        >
          <SessionContext :session="session" class="xl:sticky xl:top-5" @reset="reset" />
          <ConversationTimeline
            :session="session"
            :query-understanding="queryUnderstanding"
            :query-compiler="queryCompiler"
            :intent-transparency="intentTransparency"
            :retrieval="retrieval"
            :ranking="ranking"
            :pending-agent-turn="pendingAgentTurn"
            @animation-complete="handleAnimationComplete"
          />
          <aside class="space-y-3 xl:sticky xl:top-5">
            <QueryUnderstandingCard
              v-if="persistentQueryUnderstanding"
              card
              :progress="persistentQueryUnderstanding"
            />
            <TransparencyCard
              v-if="persistentTransparency"
              :progress="persistentTransparency"
            />
            <AgentComposer
              v-if="session.mode === 'human_as_agent' && session.status === 'waiting_for_agent'"
              ref="composer"
              :loading="loading"
              @submit="submit"
            />
            <HumanSimulatorComposer
              v-else-if="session.mode === 'human_as_simulator' && session.status === 'waiting_for_simulator'"
              :session="session"
              :loading="loading"
              @reply="replyAsSimulator"
            />
            <AutoConversationControls
              v-else-if="session.mode === 'agent_simulator' && session.status === 'waiting_for_agent'"
              :loading="loading"
              :running="autoRunning"
              :actor="autoActor"
              @step="advanceAuto"
              @toggle="toggleAuto"
            />
            <div
              v-else-if="session.status === 'initializing'"
              class="rounded-xl border bg-card p-6 text-center text-sm text-muted-foreground"
            >
              {{ session.mode === 'human_as_simulator' ? 'Preparing Agent…' : session.mode === 'agent_simulator' ? 'Preparing Agent and Simulator…' : 'Simulator is preparing the first reply…' }}
            </div>
            <div v-else class="rounded-xl border bg-card p-6 text-center text-sm text-muted-foreground">
              This session is complete. Review the target and conversation, or start a new case.
            </div>
          </aside>
        </motion.div>
      </AnimatePresence>
    </main>
  </div>
  </MotionConfig>
</template>
