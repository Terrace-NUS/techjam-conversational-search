<script setup lang="ts">
import { ref, watch } from 'vue'
import { LoaderCircle, PackageSearch } from '@lucide/vue'
import { getProduct } from '@/api'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogScrollContent,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import type { ProductSummary } from '@/types'

const props = defineProps<{ product: ProductSummary }>()
const open = ref(false)
const detail = ref(props.product)
const loading = ref(false)
const error = ref('')

watch(() => props.product, (product) => { detail.value = product })
watch(open, async (isOpen) => {
  if (!isOpen || detail.value.features !== undefined) return
  loading.value = true
  error.value = ''
  try {
    detail.value = await getProduct(props.product.parent_asin)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load product details'
  } finally {
    loading.value = false
  }
})

function priceLabel(price: ProductSummary['price']): string {
  if (price === null || price === '') return 'No price'
  const numeric = Number(price)
  return Number.isFinite(numeric) ? `$${numeric.toFixed(2)}` : String(price)
}
</script>

<template>
  <Dialog v-model:open="open">
    <DialogTrigger as-child>
      <slot />
    </DialogTrigger>
    <DialogScrollContent class="max-w-3xl sm:max-w-3xl">
      <DialogHeader>
        <DialogTitle>Product details</DialogTitle>
        <DialogDescription class="font-mono">{{ detail.parent_asin }}</DialogDescription>
      </DialogHeader>

      <Card>
        <CardContent class="space-y-6 pt-6">
          <div class="flex items-start gap-5">
            <img
              v-if="detail.thumb"
              :src="detail.thumb"
              alt=""
              class="size-28 shrink-0 rounded-xl border bg-white object-contain p-2"
              referrerpolicy="no-referrer"
            />
            <div v-else class="grid size-28 shrink-0 place-items-center rounded-xl border bg-muted text-muted-foreground">
              <PackageSearch class="size-8" />
            </div>
            <div class="min-w-0 space-y-3">
              <h3 class="text-lg font-semibold leading-7">{{ detail.title }}</h3>
              <div class="flex flex-wrap gap-x-4 gap-y-2 text-sm text-muted-foreground">
                <span class="font-medium text-foreground">{{ priceLabel(detail.price) }}</span>
                <span v-if="detail.average_rating !== null">★ {{ detail.average_rating }}</span>
                <span v-if="detail.rating_number !== null">{{ detail.rating_number }} reviews</span>
                <span v-if="detail.store">{{ detail.store }}</span>
              </div>
              <div class="flex flex-wrap gap-1.5">
                <Badge v-for="category in detail.categories" :key="category" variant="secondary">
                  {{ category }}
                </Badge>
              </div>
            </div>
          </div>

          <div v-if="loading" class="grid place-items-center py-10 text-muted-foreground">
            <LoaderCircle class="size-6 animate-spin" />
          </div>
          <p v-else-if="error" class="text-sm text-destructive">{{ error }}</p>
          <template v-else>
            <section v-if="detail.features?.length" class="space-y-2">
              <h4 class="text-sm font-semibold">Features</h4>
              <ul class="list-disc space-y-1.5 pl-5 text-sm leading-6 text-muted-foreground">
                <li v-for="feature in detail.features" :key="feature">{{ feature }}</li>
              </ul>
            </section>
            <section v-if="detail.description?.length" class="space-y-2">
              <h4 class="text-sm font-semibold">Description</h4>
              <p v-for="paragraph in detail.description" :key="paragraph" class="text-sm leading-6 text-muted-foreground">
                {{ paragraph }}
              </p>
            </section>
            <section v-if="Object.keys(detail.details ?? {}).length" class="space-y-2">
              <h4 class="text-sm font-semibold">Specifications</h4>
              <dl class="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-4 gap-y-2 text-sm">
                <template v-for="(value, key) in detail.details" :key="key">
                  <dt class="font-medium">{{ key }}</dt>
                  <dd class="text-muted-foreground">{{ value }}</dd>
                </template>
              </dl>
            </section>
          </template>
        </CardContent>
      </Card>
    </DialogScrollContent>
  </Dialog>
</template>
