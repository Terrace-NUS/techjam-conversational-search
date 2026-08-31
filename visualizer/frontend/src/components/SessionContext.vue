<script setup lang="ts">
import { computed } from 'vue'
import { RotateCcw, Target } from '@lucide/vue'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import ProductDetailDialog from './ProductDetailDialog.vue'
import type { SimulatorSession } from '@/types'

const props = defineProps<{ session: SimulatorSession }>()
defineEmits<{ reset: [] }>()

const targetProduct = computed(
  () => props.session.outcome?.target_product ?? props.session.debug_target_product,
)
const intentEntries = computed(() => Object.entries(props.session.human_context?.intent_description ?? {}))
</script>

<template>
  <aside class="space-y-3">
    <Card>
      <CardHeader class="flex-row items-start justify-between gap-3">
        <div>
          <p class="mb-1 text-xs font-medium uppercase tracking-widest text-muted-foreground">Session</p>
          <CardTitle class="font-mono text-base">{{ session.sample.sample_id }}</CardTitle>
        </div>
        <Badge variant="outline">Turn {{ session.current_turn }}/10</Badge>
      </CardHeader>
      <CardContent class="space-y-4 text-sm">
        <div class="flex flex-wrap gap-2">
          <Badge>{{ session.sample.scenario_type.replaceAll('_', ' ') }}</Badge>
          <Badge variant="secondary">{{ session.sample.difficulty_bucket }}</Badge>
          <Badge variant="outline">{{ session.sample.category_bucket }}</Badge>
          <Badge v-if="session.agent" variant="outline">{{ session.agent }} agent</Badge>
        </div>
        <template v-if="session.mode === 'human_as_agent'">
          <Separator />
          <div>
            <p class="mb-1 font-medium">Customer profile</p>
            <p class="leading-6 text-muted-foreground">{{ session.user_profile.summary }}</p>
          </div>
          <div class="flex flex-wrap gap-1.5">
            <Badge
              v-for="tag in session.user_profile.preference_tags"
              :key="tag"
              variant="secondary"
            >
              {{ tag }}
            </Badge>
          </div>
        </template>
      </CardContent>
    </Card>

    <details v-if="session.human_context" class="group rounded-xl border bg-card text-card-foreground shadow-sm">
      <summary class="cursor-pointer list-none px-4 py-3 text-sm font-medium">
        Simulator brief
        <span class="float-right text-xs text-muted-foreground group-open:hidden">Show</span>
        <span class="float-right text-xs text-muted-foreground hidden group-open:inline">Hide</span>
      </summary>
      <div class="space-y-3 border-t px-4 py-3 text-sm">
        <div class="flex flex-wrap gap-2">
          <Badge>{{ session.human_context.intent }}</Badge>
          <Badge v-if="session.human_context.override" variant="secondary">
            Override · turn {{ session.human_context.modify_turn }}
          </Badge>
        </div>
        <dl v-if="intentEntries.length" class="space-y-2">
          <div v-for="([key, value]) in intentEntries" :key="key">
            <dt class="font-medium capitalize">{{ key.replaceAll('_', ' ') }}</dt>
            <dd class="leading-5 text-muted-foreground">{{ value }}</dd>
          </div>
        </dl>
        <p v-else class="text-muted-foreground">Use the target product details to answer naturally.</p>
      </div>
    </details>

    <Card
      v-if="targetProduct"
      :class="session.outcome?.hit ? 'border-emerald-300 dark:border-emerald-800' : session.outcome ? 'border-amber-300 dark:border-amber-800' : 'border-sky-300 dark:border-sky-800'"
    >
      <CardHeader>
        <div class="flex items-center gap-2">
          <Target class="size-4" />
          <CardTitle class="text-base">
            {{ session.outcome ? 'Hidden target revealed' : session.mode === 'human_as_simulator' ? 'Target product' : 'Debug target' }}
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent class="space-y-2 text-sm">
        <ProductDetailDialog :product="targetProduct">
          <button type="button" class="flex w-full items-start gap-3 rounded-lg text-left outline-none focus-visible:ring-2 focus-visible:ring-ring">
            <img
              v-if="targetProduct.thumb"
              :src="targetProduct.thumb"
              alt=""
              class="size-14 shrink-0 rounded-lg border bg-white object-contain p-1"
              referrerpolicy="no-referrer"
            />
            <div class="min-w-0">
              <p class="font-medium leading-5">{{ targetProduct.title }}</p>
              <p class="mt-1 font-mono text-xs text-muted-foreground">
                {{ targetProduct.parent_asin }}
              </p>
            </div>
          </button>
        </ProductDetailDialog>
        <p v-if="session.outcome?.hit" class="font-medium text-emerald-700 dark:text-emerald-400">
          Hit at rank {{ session.outcome.best_rank }} on turn {{ session.outcome.first_hit_turn }}.
        </p>
        <p v-else-if="session.outcome" class="font-medium text-amber-700 dark:text-amber-400">No hit within ten turns.</p>
      </CardContent>
    </Card>

    <Button class="w-full" variant="outline" @click="$emit('reset')">
      <RotateCcw class="size-4" />
      New session
    </Button>
  </aside>
</template>
