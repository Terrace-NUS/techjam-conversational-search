<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'
import { CircleAlert, Moon, Palette, Sparkles, Sun } from '@lucide/vue'
import { AnimatePresence, motion, MotionConfig, useAnimate } from 'motion-v'
import { createSession, getDatasets, getSamples, initializeSession, submitAgentTurn } from './api'
import AgentComposer from './components/AgentComposer.vue'
import ConversationTimeline from './components/ConversationTimeline.vue'
import SessionContext from './components/SessionContext.vue'
import SessionSetup from './components/SessionSetup.vue'
import { Badge } from '@/components/ui/badge'
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
  ReplyModel,
  SampleSummary,
  SimulatorSession,
} from './types'

type ThemePalette = 'default' | 'claude'
const SIMULATOR_REPLY_DELAY_MS = 2000

const samples = ref<SampleSummary[]>([])
const datasets = ref<DatasetOption[]>([])
const session = ref<SimulatorSession | null>(null)
const loading = ref(false)
const error = ref('')
const composer = ref<InstanceType<typeof AgentComposer> | null>(null)
const darkMode = ref(document.documentElement.classList.contains('dark'))
const themePalette = ref<ThemePalette>(
  document.documentElement.classList.contains('theme-claude') ? 'claude' : 'default',
)
const [appScope, animateApp] = useAnimate()

watch(themePalette, (palette) => {
  document.documentElement.classList.toggle('theme-claude', palette === 'claude')
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

async function start(
  sampleId: string,
  dataset: string,
  replyModel: ReplyModel,
  debug: boolean,
) {
  loading.value = true
  error.value = ''
  try {
    const created = await createSession(sampleId, dataset, replyModel, debug)
    session.value = created
    await nextTick()
    const initialized = await initializeSession(created.id)
    if (session.value?.id !== created.id) return
    session.value = initialized
    if (session.value?.status === 'error') {
      throw new Error(session.value.initialization_error ?? 'Could not generate first reply')
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not start session'
  } finally {
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
          agent_message: input.message,
          ask_attribute: input.ask_attribute,
          recommendations: products,
          hit_rank: null,
        },
      ],
    }
    await nextTick()
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    const replyDelay = new Promise((resolve) => setTimeout(resolve, SIMULATOR_REPLY_DELAY_MS))
    const nextSession = await response
    if (nextSession.status === 'waiting_for_agent') await replyDelay
    session.value = nextSession
    composer.value?.clear()
    await nextTick()
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
  } catch (cause) {
    session.value = previousSession
    error.value = cause instanceof Error ? cause.message : 'Could not submit turn'
  } finally {
    loading.value = false
  }
}

function reset() {
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
            <h1 class="font-semibold tracking-tight">Conversational Search Lab</h1>
            <p class="text-xs text-muted-foreground">Simulator visualizer</p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <Badge variant="outline" class="hidden sm:inline-flex">Human as Agent</Badge>
          <Select v-model="themePalette">
            <SelectTrigger class="w-[8.5rem]" aria-label="Theme palette">
              <Palette class="size-4" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">Default</SelectItem>
              <SelectItem value="claude">Claude</SelectItem>
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
          <ConversationTimeline :session="session" />
          <AgentComposer
            v-if="session.status === 'waiting_for_agent'"
            ref="composer"
            class="xl:sticky xl:top-5"
            :loading="loading"
            @submit="submit"
          />
          <div
            v-else-if="session.status === 'initializing'"
            class="rounded-xl border bg-card p-6 text-center text-sm text-muted-foreground"
          >
            Simulator is preparing the first reply…
          </div>
          <div v-else class="rounded-xl border bg-card p-6 text-center text-sm text-muted-foreground">
            This session is complete. Review the target and conversation, or start a new case.
          </div>
        </motion.div>
      </AnimatePresence>
    </main>
  </div>
  </MotionConfig>
</template>
