<script setup lang="ts">
import { Bot, CheckCircle2, UserRound } from '@lucide/vue'
import { motion } from 'motion-v'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { SimulatorSession } from '@/types'

defineProps<{ session: SimulatorSession }>()
</script>

<template>
  <Card class="min-h-[32rem]">
    <CardHeader class="border-b">
      <div class="flex items-center justify-between gap-3">
        <CardTitle class="text-base">Conversation</CardTitle>
        <Badge v-if="session.status === 'initializing'" variant="secondary">Simulator replying…</Badge>
        <Badge v-else-if="session.status === 'waiting_for_agent'" variant="secondary">Your turn</Badge>
        <Badge v-else :variant="session.status === 'hit' ? 'default' : 'outline'">
          {{ session.status === 'hit' ? 'Target found' : 'Session complete' }}
        </Badge>
      </div>
    </CardHeader>
    <CardContent class="space-y-7 py-6">
      <div v-for="(turn, index) in session.turns" :key="index" class="space-y-4">
        <motion.div
          class="flex gap-3"
          :initial="{ opacity: 0, x: -14 }"
          :animate="{ opacity: 1, x: 0 }"
          :transition="{ duration: 0.24, ease: 'easeOut' }"
        >
          <div class="grid size-8 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
            <UserRound class="size-4" />
          </div>
          <div class="min-w-0 flex-1">
            <p class="mb-1 text-xs font-medium text-muted-foreground">Simulator · Turn {{ index + 1 }}</p>
            <p class="rounded-xl rounded-tl-sm bg-muted px-4 py-3 text-sm leading-6">
              {{ turn.user_message }}
            </p>
            <p v-if="turn.user_message_original" class="mt-1.5 text-xs leading-5 text-muted-foreground">
              Original: {{ turn.user_message_original }}
            </p>
          </div>
        </motion.div>

        <motion.div
          class="ml-8 flex flex-row-reverse gap-3"
          :initial="{ opacity: 0, x: 14 }"
          :animate="{ opacity: 1, x: 0 }"
          :transition="{ duration: 0.26, delay: 0.08, ease: 'easeOut' }"
        >
          <div class="grid size-8 shrink-0 place-items-center rounded-full border bg-card">
            <Bot class="size-4" />
          </div>
          <div class="flex min-w-0 flex-1 flex-col items-end space-y-2">
            <div class="flex flex-wrap items-center justify-end gap-2">
              <p class="text-xs font-medium text-muted-foreground">You · Agent</p>
              <Badge v-if="turn.hit_rank" class="bg-emerald-600">
                <CheckCircle2 class="size-3" /> rank {{ turn.hit_rank }}
              </Badge>
            </div>
            <div
              v-if="turn.agent_message.trim() || turn.ask_attribute"
              class="max-w-[90%] space-y-2 rounded-xl rounded-tr-sm border bg-card px-4 py-3 text-left text-sm leading-6"
            >
              <p v-if="turn.agent_message.trim()">{{ turn.agent_message }}</p>
              <Badge v-if="turn.ask_attribute" variant="outline">ask: {{ turn.ask_attribute }}</Badge>
            </div>
            <div v-if="turn.recommendations.length" class="flex flex-wrap justify-end gap-1.5">
              <Badge
                v-for="(product, productIndex) in turn.recommendations"
                :key="product.parent_asin"
                variant="secondary"
                :title="product.title"
              >
                <img
                  v-if="product.thumb"
                  :src="product.thumb"
                  alt=""
                  class="size-5 rounded-sm bg-white object-contain"
                  loading="lazy"
                  referrerpolicy="no-referrer"
                />
                #{{ productIndex + 1 }} {{ product.parent_asin }}
              </Badge>
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
            {{ session.current_user_message }}
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
          <p class="mb-1 text-xs font-medium text-muted-foreground">Simulator · Turn 1</p>
          <p class="rounded-xl rounded-tl-sm bg-muted px-4 py-3 text-sm text-muted-foreground">
            Generating reply…
          </p>
        </div>
      </motion.div>
    </CardContent>
  </Card>
</template>
