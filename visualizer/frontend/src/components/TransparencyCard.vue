<script setup lang="ts">
import { computed } from 'vue'
import { Gauge } from '@lucide/vue'
import { nonlinearScore } from '@/lib/utils'
import type { IntentTransparencyEvent } from '@/types'

const props = defineProps<{ progress: IntentTransparencyEvent }>()

const value = computed(() => (
  props.progress.estimate?.transparency ?? props.progress.applied_transparency ?? null
))
const percentage = computed(() => value.value == null ? null : Math.round(nonlinearScore(value.value) * 100))
</script>

<template>
  <div class="rounded-xl border bg-card px-6 py-5 shadow-sm">
    <div class="flex items-center gap-2 text-sm font-medium text-foreground">
      <Gauge class="size-4 text-muted-foreground" />
      Intent Transparency
      <span class="ml-auto text-xs font-normal text-muted-foreground">Turn {{ progress.turn }}</span>
    </div>
    <div class="mt-5 flex items-end justify-between gap-4">
      <Transition name="transparency-value" mode="out-in" appear>
        <p :key="percentage ?? 'unavailable'" class="text-4xl font-semibold leading-none text-foreground">
          {{ percentage === null ? '—' : `${percentage}%` }}
        </p>
      </Transition>
    </div>
    <div class="mt-4 h-2 overflow-hidden rounded-full bg-muted">
      <div
        class="h-full rounded-full bg-primary transition-[width] duration-700 ease-out"
        :style="{ width: `${percentage ?? 0}%` }"
      />
    </div>
    <div class="mt-1.5 flex justify-between text-xs text-muted-foreground">
      <span>Broad</span>
      <span>Focused</span>
    </div>
  </div>
</template>

<style scoped>
.transparency-value-enter-active,
.transparency-value-leave-active {
  transition: opacity 220ms ease, transform 220ms ease;
}

.transparency-value-enter-from {
  opacity: 0;
  transform: translateY(5px);
}

.transparency-value-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
