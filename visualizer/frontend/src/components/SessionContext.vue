<script setup lang="ts">
import { RotateCcw, Target } from '@lucide/vue'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import type { SimulatorSession } from '@/types'

defineProps<{ session: SimulatorSession }>()
defineEmits<{ reset: [] }>()
</script>

<template>
  <aside class="space-y-4">
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
        </div>
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
      </CardContent>
    </Card>

    <Card v-if="session.outcome" :class="session.outcome.hit ? 'border-emerald-300 dark:border-emerald-800' : 'border-amber-300 dark:border-amber-800'">
      <CardHeader>
        <div class="flex items-center gap-2">
          <Target class="size-4" />
          <CardTitle class="text-base">Hidden target revealed</CardTitle>
        </div>
      </CardHeader>
      <CardContent class="space-y-2 text-sm">
        <div class="flex items-start gap-3">
          <img
            v-if="session.outcome.target_product.thumb"
            :src="session.outcome.target_product.thumb"
            alt=""
            class="size-14 shrink-0 rounded-lg border bg-white object-contain p-1"
            referrerpolicy="no-referrer"
          />
          <div class="min-w-0">
            <p class="font-medium leading-5">{{ session.outcome.target_product.title }}</p>
            <p class="mt-1 font-mono text-xs text-muted-foreground">
              {{ session.outcome.target_product.parent_asin }}
            </p>
          </div>
        </div>
        <p v-if="session.outcome.hit" class="font-medium text-emerald-700 dark:text-emerald-400">
          Hit at rank {{ session.outcome.best_rank }} on turn {{ session.outcome.first_hit_turn }}.
        </p>
        <p v-else class="font-medium text-amber-700 dark:text-amber-400">No hit within ten turns.</p>
      </CardContent>
    </Card>

    <Button class="w-full" variant="outline" @click="$emit('reset')">
      <RotateCcw class="size-4" />
      New session
    </Button>
  </aside>
</template>
