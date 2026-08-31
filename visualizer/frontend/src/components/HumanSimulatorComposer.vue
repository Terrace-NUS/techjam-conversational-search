<script setup lang="ts">
import { ref, watch } from 'vue'
import { Send, Sparkles } from '@lucide/vue'
import { rewriteMessage } from '@/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import type { SimulatorSession } from '@/types'

const props = defineProps<{ session: SimulatorSession; loading: boolean }>()
const emit = defineEmits<{
  reply: [message: string]
}>()

const message = ref('')
const rewriting = ref(false)
const error = ref('')

watch(
  () => props.session.turns.length,
  () => {
    message.value = ''
    error.value = ''
  },
)

async function rewrite() {
  if (!message.value.trim()) return
  rewriting.value = true
  error.value = ''
  try {
    message.value = (await rewriteMessage(message.value.trim())).message
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not rewrite message'
  } finally {
    rewriting.value = false
  }
}
</script>

<template>
  <Card>
    <CardHeader>
      <CardTitle class="text-base">Reply as Simulator</CardTitle>
    </CardHeader>
    <CardContent v-if="session.status === 'waiting_for_simulator'" class="space-y-4">
      <label class="grid gap-2 text-sm font-medium">
        Customer reply
        <Textarea
          v-model="message"
          rows="5"
          placeholder="Write what the customer should say next…"
          :disabled="loading || rewriting"
        />
      </label>
      <p v-if="error" class="text-xs text-red-600 dark:text-red-400">{{ error }}</p>
      <div class="grid grid-cols-2 gap-2">
        <Button
          variant="outline"
          :disabled="loading || rewriting || !message.trim()"
          @click="rewrite"
        >
          <Sparkles class="size-4" />
          {{ rewriting ? 'Rewriting…' : 'Rewrite with DeepSeek' }}
        </Button>
        <Button :disabled="loading || rewriting || !message.trim()" @click="emit('reply', message.trim())">
          <Send class="size-4" />
          {{ loading ? 'Agent is responding…' : 'Send to Agent' }}
        </Button>
      </div>
    </CardContent>
  </Card>
</template>
