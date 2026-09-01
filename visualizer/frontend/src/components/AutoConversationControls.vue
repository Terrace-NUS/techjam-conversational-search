<script setup lang="ts">
import { Bot, Pause, Play, UserRound } from '@lucide/vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

defineProps<{
  loading: boolean
  running: boolean
  actor: 'agent' | 'simulator'
}>()

defineEmits<{
  toggle: []
  step: []
}>()
</script>

<template>
  <Card>
    <CardHeader>
      <CardTitle class="text-base">Agent ↔ Simulator</CardTitle>
      <CardDescription>
        Run continuously, or execute one side at a time.
      </CardDescription>
    </CardHeader>
    <CardContent class="grid gap-3">
      <Button
        :variant="running ? 'destructive' : 'default'"
        :disabled="loading && !running"
        @click="$emit('toggle')"
      >
        <Pause v-if="running" class="size-4" />
        <Play v-else class="size-4" />
        {{ running ? 'Pause' : 'Auto run' }}
      </Button>
      <Button variant="outline" :disabled="loading || running" @click="$emit('step')">
        <Bot v-if="actor === 'agent'" class="size-4" />
        <UserRound v-else class="size-4" />
        {{ loading ? `Running ${actor}…` : `Run ${actor === 'agent' ? 'Agent' : 'Simulator'}` }}
      </Button>
    </CardContent>
  </Card>
</template>
