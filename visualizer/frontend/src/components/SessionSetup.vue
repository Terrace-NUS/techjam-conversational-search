<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { Play } from '@lucide/vue'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { DatasetOption, ReplyModel, SampleSummary } from '@/types'

const props = defineProps<{
  samples: SampleSummary[]
  datasets: DatasetOption[]
  loading: boolean
}>()

const emit = defineEmits<{
  start: [sampleId: string, dataset: string, replyModel: ReplyModel, debug: boolean]
  datasetChange: [dataset: string]
}>()

const scenario = ref('all')
const selectedId = ref('')
const dataset = ref('')
const replyModel = ref<ReplyModel>('deepseek')
const debug = ref(false)

const scenarios = computed(() => [
  'all',
  ...new Set(props.samples.map((sample) => sample.scenario_type)),
])

const filteredSamples = computed(() =>
  scenario.value === 'all'
    ? props.samples
    : props.samples.filter((sample) => sample.scenario_type === scenario.value),
)

watch(
  () => props.datasets,
  (datasets) => {
    if (!datasets.some((item) => item.id === dataset.value)) {
      dataset.value = datasets.find((item) => item.default)?.id ?? datasets[0]?.id ?? ''
    }
  },
  { immediate: true },
)

watch(dataset, (value, previous) => {
  if (value && previous) emit('datasetChange', value)
})

watch(
  filteredSamples,
  (samples) => {
    if (!samples.some((sample) => sample.sample_id === selectedId.value)) {
      selectedId.value = samples[0]?.sample_id ?? ''
    }
  },
  { immediate: true },
)
</script>

<template>
  <Card class="w-full max-w-2xl border-0 shadow-xl shadow-slate-200/60 dark:shadow-black/30">
    <CardHeader class="space-y-3">
      <div class="flex items-center gap-2">
        <Badge variant="secondary">Human as Agent</Badge>
        <Badge variant="outline">{{ replyModel === 'deepseek' ? 'DeepSeek replies' : 'Template replies' }}</Badge>
      </div>
      <CardTitle class="text-2xl">Start a product-guessing session</CardTitle>
      <CardDescription class="max-w-xl leading-6">
        You will see exactly what an Agent sees. Ask one structured question, rank up to
        ten products, and find the hidden target within ten turns.
      </CardDescription>
    </CardHeader>
    <CardContent class="space-y-5">
      <div class="grid gap-4 sm:grid-cols-2">
        <div class="grid gap-2 text-sm font-medium">
          <label for="dataset-select">Dataset</label>
          <Select v-model="dataset">
            <SelectTrigger id="dataset-select" class="w-full">
              <SelectValue placeholder="Choose a dataset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem v-for="item in datasets" :key="item.id" :value="item.id">
                {{ item.label }} · {{ item.sample_count }}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div class="grid gap-2 text-sm font-medium">
          <label for="reply-model-select">Reply model</label>
          <Select v-model="replyModel">
            <SelectTrigger id="reply-model-select" class="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="template">Template</SelectItem>
              <SelectItem value="deepseek">DeepSeek</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div class="grid gap-2 text-sm font-medium">
          <label for="scenario-select">Scenario</label>
          <Select v-model="scenario">
            <SelectTrigger id="scenario-select" class="w-full">
              <SelectValue placeholder="Choose a scenario" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem v-for="item in scenarios" :key="item" :value="item">
                {{ item === 'all' ? 'All scenarios' : item.replaceAll('_', ' ') }}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div class="grid gap-2 text-sm font-medium">
          <label for="sample-select">Public case</label>
          <Select v-model="selectedId">
            <SelectTrigger id="sample-select" class="w-full">
              <SelectValue placeholder="Choose a public case" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem
                v-for="sample in filteredSamples"
                :key="sample.sample_id"
                :value="sample.sample_id"
              >
                {{ sample.sample_id }} · {{ sample.difficulty_bucket }}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div class="rounded-lg border bg-muted/40 p-4 text-sm text-muted-foreground">
        <label class="flex cursor-pointer items-start gap-3">
          <input v-model="debug" type="checkbox" class="mt-0.5 size-4 accent-primary" />
          <span>
            <span class="font-medium text-foreground">Debug mode</span><br />
            Show the target product and DeepSeek's canonical input during the session.
          </span>
        </label>
      </div>

      <Button
        class="w-full"
        size="lg"
        :disabled="loading || !selectedId || !dataset"
        @click="emit('start', selectedId, dataset, replyModel, debug)"
      >
        <Play class="size-4" />
        {{ loading ? 'Starting…' : 'Start session' }}
      </Button>
    </CardContent>
  </Card>
</template>
