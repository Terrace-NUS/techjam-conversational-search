<script setup lang="ts">
import { Pause, Play, StepForward } from '@lucide/vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

defineProps<{
  loading: boolean
  running: boolean
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
        Run continuously, or advance exactly one complete turn.
      </CardDescription>
    </CardHeader>
    <CardContent class="grid gap-3">
      <Button :variant="running ? 'destructive' : 'default'" @click="$emit('toggle')">
        <Pause v-if="running" class="size-4" />
        <Play v-else class="size-4" />
        {{ running ? 'Pause' : 'Auto run' }}
      </Button>
      <Button variant="outline" :disabled="loading || running" @click="$emit('step')">
        <StepForward class="size-4" />
        {{ loading ? 'Running turn…' : 'Next turn' }}
      </Button>
    </CardContent>
  </Card>
</template>
