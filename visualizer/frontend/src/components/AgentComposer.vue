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
const inputMode = ref<'message' | 'attribute'>('message')
const askAttribute = ref<AskAttribute>('category')
const recommendations = ref<ProductSummary[]>([])

function submit() {
  emit(
    'submit',
    {
      message: inputMode.value === 'message' ? message.value : '',
      ask_attribute: inputMode.value === 'attribute' ? askAttribute.value : null,
      recommendations: recommendations.value.map((product) => product.parent_asin),
    },
    [...recommendations.value],
  )
}

function clear() {
  message.value = ''
  askAttribute.value = 'category'
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
      <div class="grid grid-cols-2 rounded-md border p-1" role="group" aria-label="Query input mode">
        <button
          type="button"
          :aria-pressed="inputMode === 'message'"
          :class="['h-9 rounded-sm px-3 text-sm font-medium transition-colors', inputMode === 'message' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:bg-muted']"
          :disabled="loading"
          @click="inputMode = 'message'"
        >
          Message
        </button>
        <button
          type="button"
          :aria-pressed="inputMode === 'attribute'"
          :class="['h-9 rounded-sm px-3 text-sm font-medium transition-colors', inputMode === 'attribute' ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:bg-muted']"
          :disabled="loading"
          @click="inputMode = 'attribute'"
        >
          Attribute
        </button>
      </div>

      <label v-if="inputMode === 'message'" class="grid gap-2 text-sm font-medium">
        Message
        <Textarea
          v-model="message"
          rows="3"
          placeholder="Ask a useful clarification or explain your recommendations…"
          :disabled="loading"
        />
      </label>

      <div v-else class="grid gap-2 text-sm font-medium">
        <label for="ask-attribute-select">Ask attribute</label>
        <Select v-model="askAttribute" :disabled="loading">
          <SelectTrigger id="ask-attribute-select" class="w-full">
            <SelectValue placeholder="Choose an attribute" />
          </SelectTrigger>
          <SelectContent>
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
