<script setup lang="ts">
import { Bot, CheckCircle2, UserRound } from '@lucide/vue'
import { motion } from 'motion-v'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import ProductDetailDialog from './ProductDetailDialog.vue'
import type { SimulatorSession } from '@/types'

defineProps<{ session: SimulatorSession }>()
</script>

<template>
  <Card class="min-h-[32rem]">
    <CardHeader class="border-b">
      <div class="flex items-center justify-between gap-3">
        <CardTitle class="text-base">Conversation</CardTitle>
        <Badge v-if="session.status === 'initializing'" variant="secondary">
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
              {{ turn.user_message }}
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
                {{ session.mode === 'human_as_agent' ? 'You · Agent' : `${session.agent?.toUpperCase()} · Agent` }}
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
            <div
              v-if="turn.agent_message.trim() || turn.ask_attribute"
              :class="['max-w-[90%] space-y-2 rounded-xl border bg-card px-4 py-3 text-left text-sm leading-6', session.mode === 'human_as_simulator' ? 'rounded-tl-sm' : 'rounded-tr-sm']"
            >
              <p v-if="turn.agent_message.trim()">{{ turn.agent_message }}</p>
              <Badge v-if="turn.ask_attribute" variant="outline">ask: {{ turn.ask_attribute }}</Badge>
            </div>
            <div :class="['flex flex-wrap gap-1.5', session.mode !== 'human_as_simulator' && 'justify-end']" v-if="turn.recommendations.length">
              <ProductDetailDialog
                v-for="(product, productIndex) in turn.recommendations"
                :key="product.parent_asin"
                :product="product"
              >
                <Badge
                  as="button"
                  type="button"
                  variant="secondary"
                  class="cursor-pointer"
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
                  <span v-if="session.debug" class="font-mono opacity-70">
                    · {{ turn.recommendation_scores[product.parent_asin] === null || turn.recommendation_scores[product.parent_asin] === undefined
                      ? 'score —'
                      : `score ${turn.recommendation_scores[product.parent_asin]?.toFixed(3)}` }}
                  </span>
                </Badge>
              </ProductDetailDialog>
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
</template>
