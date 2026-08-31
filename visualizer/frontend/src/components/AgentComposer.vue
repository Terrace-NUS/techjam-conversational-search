<script setup lang="ts">
import { ref } from 'vue'
import { Send } from '@lucide/vue'
import ProductPicker from './ProductPicker.vue'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { ASK_ATTRIBUTES, type AgentTurnInput, type AskAttribute, type ProductSummary } from '@/types'

defineProps<{ loading: boolean }>()

const emit = defineEmits<{
  submit: [input: AgentTurnInput, products: ProductSummary[]]
}>()

const message = ref('')
const askAttribute = ref<AskAttribute | 'none'>('none')
const recommendations = ref<ProductSummary[]>([])

function submit() {
  emit(
    'submit',
    {
      message: message.value,
      ask_attribute: askAttribute.value === 'none' ? null : askAttribute.value,
      recommendations: recommendations.value.map((product) => product.parent_asin),
    },
    [...recommendations.value],
  )
}

function clear() {
  message.value = ''
  askAttribute.value = 'none'
  recommendations.value = []
}

defineExpose({ clear })
</script>

<template>
  <Card>
    <CardHeader>
      <CardTitle class="text-base">Compose Agent response</CardTitle>
    </CardHeader>
    <CardContent class="space-y-5">
      <label class="grid gap-2 text-sm font-medium">
        Message <span class="font-normal text-muted-foreground">(optional)</span>
        <Textarea
          v-model="message"
          rows="3"
          placeholder="Ask a useful clarification or explain your recommendations…"
          :disabled="loading"
        />
      </label>

      <div class="grid gap-2 text-sm font-medium">
        <label for="ask-attribute-select">Ask attribute</label>
        <Select v-model="askAttribute" :disabled="loading">
          <SelectTrigger id="ask-attribute-select" class="w-full">
            <SelectValue placeholder="Choose an attribute" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            <SelectItem v-for="attribute in ASK_ATTRIBUTES" :key="attribute" :value="attribute">
              {{ attribute.replaceAll('_', ' ') }}
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <ProductPicker v-model="recommendations" :disabled="loading" />

      <Button class="w-full" size="lg" :disabled="loading" @click="submit">
        <Send class="size-4" />
        {{ loading ? 'Simulator is responding…' : 'Submit turn' }}
      </Button>
    </CardContent>
  </Card>
</template>
