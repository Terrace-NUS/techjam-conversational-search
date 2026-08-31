<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import {
  ChevronDown,
  ChevronUp,
  LoaderCircle,
  PackageSearch,
  Plus,
  RotateCcw,
  Search,
  X,
} from '@lucide/vue'
import { getCatalogFilters, searchCatalog } from '@/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import ProductDetailDialog from './ProductDetailDialog.vue'
import type { CatalogFilters, CatalogSearchInput, ProductSummary } from '@/types'

const PAGE_SIZE = 40

interface HighlightSegment {
  text: string
  matched: boolean
}

const props = defineProps<{
  modelValue: ProductSummary[]
  disabled?: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [products: ProductSummary[]]
}>()

const open = ref(false)
const query = ref('')
const minPrice = ref('')
const maxPrice = ref('')
const minRating = ref('')
const minRatingCount = ref('')
const filters = ref<CatalogFilters | null>(null)
const results = ref<ProductSummary[]>([])
const loading = ref(false)
const loadingMore = ref(false)
const hasMore = ref(false)
const error = ref('')
let timer: ReturnType<typeof setTimeout> | undefined
let controller: AbortController | undefined

const selectedIds = computed(() => new Set(props.modelValue.map((product) => product.parent_asin)))

function numberValue(value: string): number | undefined {
  if (value === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

async function loadFilters() {
  if (filters.value) return
  try {
    filters.value = await getCatalogFilters()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : 'Could not load catalog filters'
  }
}

function catalogInput(offset = 0): CatalogSearchInput {
  return {
    q: query.value.trim(),
    min_price: numberValue(minPrice.value),
    max_price: numberValue(maxPrice.value),
    min_rating: numberValue(minRating.value),
    min_rating_count: numberValue(minRatingCount.value),
    limit: PAGE_SIZE + 1,
    offset,
  }
}

async function refreshResults() {
  controller?.abort()
  const currentController = new AbortController()
  controller = currentController
  loading.value = true
  loadingMore.value = false
  hasMore.value = false
  error.value = ''
  try {
    const page = await searchCatalog(catalogInput(), currentController.signal)
    results.value = page.slice(0, PAGE_SIZE)
    hasMore.value = page.length > PAGE_SIZE
  } catch (cause) {
    if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
      error.value = cause instanceof Error ? cause.message : 'Catalog search failed'
    }
  } finally {
    if (!currentController.signal.aborted) loading.value = false
  }
}

async function loadMore() {
  if (loading.value || loadingMore.value || !hasMore.value) return
  controller?.abort()
  const currentController = new AbortController()
  controller = currentController
  loadingMore.value = true
  error.value = ''
  try {
    const page = await searchCatalog(catalogInput(results.value.length), currentController.signal)
    results.value = [...results.value, ...page.slice(0, PAGE_SIZE)]
    hasMore.value = page.length > PAGE_SIZE
  } catch (cause) {
    if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
      error.value = cause instanceof Error ? cause.message : 'Could not load more products'
    }
  } finally {
    if (!currentController.signal.aborted) loadingMore.value = false
  }
}

function scheduleRefresh(delay = 180) {
  clearTimeout(timer)
  if (!open.value) return
  timer = setTimeout(refreshResults, delay)
}

watch(open, (isOpen) => {
  if (!isOpen) {
    controller?.abort()
    return
  }
  void loadFilters()
  scheduleRefresh(0)
})

watch([query, minPrice, maxPrice, minRating, minRatingCount], () => scheduleRefresh())

onBeforeUnmount(() => {
  clearTimeout(timer)
  controller?.abort()
})

function resetFilters() {
  query.value = ''
  minPrice.value = ''
  maxPrice.value = ''
  minRating.value = ''
  minRatingCount.value = ''
}

function add(product: ProductSummary) {
  if (props.modelValue.length >= 10 || selectedIds.value.has(product.parent_asin)) return
  emit('update:modelValue', [...props.modelValue, product])
}

function removeById(parentAsin: string) {
  emit('update:modelValue', props.modelValue.filter((product) => product.parent_asin !== parentAsin))
}

function remove(index: number) {
  emit('update:modelValue', props.modelValue.filter((_, itemIndex) => itemIndex !== index))
}

function move(index: number, direction: -1 | 1) {
  const nextIndex = index + direction
  if (nextIndex < 0 || nextIndex >= props.modelValue.length) return
  const next = [...props.modelValue]
  ;[next[index], next[nextIndex]] = [next[nextIndex], next[index]]
  emit('update:modelValue', next)
}

function priceLabel(price: ProductSummary['price']): string {
  if (price === null || price === '') return 'No price'
  const numeric = Number(price)
  return Number.isFinite(numeric) ? `$${numeric.toFixed(2)}` : String(price)
}

function highlight(value: string): HighlightSegment[] {
  const text = value.toLocaleLowerCase()
  const matched = new Set<number>()
  for (const term of query.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)) {
    const positions: number[] = []
    let position = -1
    for (const character of term) {
      position = text.indexOf(character, position + 1)
      if (position < 0) break
      positions.push(position)
    }
    if (positions.length === term.length) positions.forEach((index) => matched.add(index))
  }
  if (!matched.size) return [{ text: value, matched: false }]

  const segments: HighlightSegment[] = []
  for (let index = 0; index < value.length; index += 1) {
    const isMatched = matched.has(index)
    const previous = segments.at(-1)
    if (previous?.matched === isMatched) previous.text += value[index]
    else segments.push({ text: value[index], matched: isMatched })
  }
  return segments
}

function matchingFeatures(product: ProductSummary): Array<{ feature: string; index: number }> {
  return (product.features ?? [])
    .map((feature, index) => ({ feature, index }))
    .filter(({ feature }) => highlight(feature).some((segment) => segment.matched))
}
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center justify-between gap-2">
      <span class="text-sm font-medium">Ranked recommendations</span>
      <span class="text-xs text-muted-foreground">{{ modelValue.length }}/10</span>
    </div>

    <Dialog v-model:open="open">
      <DialogTrigger as-child>
        <Button type="button" class="w-full" variant="outline" :disabled="disabled">
          <PackageSearch class="size-4" />
          Browse catalog
        </Button>
      </DialogTrigger>

      <DialogContent
        class="grid h-[min(820px,calc(100vh-2rem))] max-w-[min(1100px,calc(100%-2rem))] grid-rows-[auto_auto_minmax(0,1fr)] gap-0 overflow-hidden p-0 sm:max-w-[min(1100px,calc(100%-2rem))]"
      >
        <DialogHeader class="border-b px-6 py-5 pr-12">
          <DialogTitle>Choose products</DialogTitle>
          <DialogDescription>
            Results update automatically. Search matches typed characters in order, like VS Code.
          </DialogDescription>
        </DialogHeader>

        <div class="space-y-4 border-b bg-muted/25 px-6 py-4">
          <div class="relative">
            <Search class="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              v-model="query"
              class="pl-9"
              placeholder="Search title, store, feature, or parent ASIN…"
              autofocus
            />
          </div>

          <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <label class="grid gap-1.5 text-xs font-medium text-muted-foreground">
              Min price
              <Input
                v-model="minPrice"
                type="number"
                min="0"
                step="0.01"
                :placeholder="String(filters?.price.min ?? 'Any')"
              />
            </label>
            <label class="grid gap-1.5 text-xs font-medium text-muted-foreground">
              Max price
              <Input
                v-model="maxPrice"
                type="number"
                min="0"
                step="0.01"
                :placeholder="String(filters?.price.max ?? 'Any')"
              />
            </label>
            <label class="grid gap-1.5 text-xs font-medium text-muted-foreground">
              Min rating
              <Input v-model="minRating" type="number" min="0" max="5" step="0.1" placeholder="Any" />
            </label>
            <label class="grid gap-1.5 text-xs font-medium text-muted-foreground">
              Min reviews
              <Input v-model="minRatingCount" type="number" min="0" step="1" placeholder="Any" />
            </label>
          </div>

          <div class="flex items-center justify-between gap-3 text-xs text-muted-foreground">
            <span>{{ loading ? 'Updating…' : `${results.length} products loaded` }}</span>
            <Button type="button" size="sm" variant="ghost" @click="resetFilters">
              <RotateCcw class="size-3.5" />
              Reset filters
            </Button>
          </div>
        </div>

        <div class="min-h-0 overflow-y-auto px-6 py-4">
          <div v-if="loading && !results.length" class="grid h-full place-items-center text-muted-foreground">
            <LoaderCircle class="size-6 animate-spin" />
          </div>
          <div v-else-if="error && !results.length" class="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            {{ error }}
          </div>
          <div v-else-if="!results.length" class="grid h-full place-items-center text-sm text-muted-foreground">
            No products match these conditions.
          </div>
          <div v-else class="space-y-4">
            <div class="grid items-stretch gap-3 lg:grid-cols-2">
              <article
              v-for="product in results"
              :key="product.parent_asin"
              class="h-full min-w-0 rounded-xl border bg-card p-4 shadow-xs"
            >
              <div class="flex items-start gap-3">
                <ProductDetailDialog :product="product">
                  <button type="button" class="flex min-w-0 flex-1 items-start gap-3 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring">
                    <img
                      v-if="product.thumb"
                      :src="product.thumb"
                      alt=""
                      class="size-14 shrink-0 rounded-lg border bg-white object-contain p-1"
                      loading="lazy"
                      decoding="async"
                      referrerpolicy="no-referrer"
                    />
                    <div v-else class="grid size-14 shrink-0 place-items-center rounded-lg border bg-muted text-muted-foreground">
                      <PackageSearch class="size-5" />
                    </div>
                    <div class="min-w-0 flex-1 space-y-2">
                  <div>
                    <p class="line-clamp-2 text-sm font-medium leading-5">
                      <template v-for="(segment, index) in highlight(product.title)" :key="index">
                        <mark v-if="segment.matched" class="rounded-sm bg-yellow-200 text-inherit dark:bg-yellow-500/40">{{ segment.text }}</mark>
                        <span v-else>{{ segment.text }}</span>
                      </template>
                    </p>
                    <p class="mt-1 font-mono text-[11px] text-muted-foreground">
                      <template v-for="(segment, index) in highlight(product.parent_asin)" :key="index">
                        <mark v-if="segment.matched" class="rounded-sm bg-yellow-200 text-inherit dark:bg-yellow-500/40">{{ segment.text }}</mark>
                        <span v-else>{{ segment.text }}</span>
                      </template>
                    </p>
                  </div>
                  <div class="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    <span>{{ priceLabel(product.price) }}</span>
                    <span v-if="product.average_rating !== null">★ {{ product.average_rating }}</span>
                    <span v-if="product.rating_number !== null">{{ product.rating_number }} reviews</span>
                    <span v-if="product.store" class="truncate">
                      <template v-for="(segment, index) in highlight(product.store)" :key="index">
                        <mark v-if="segment.matched" class="rounded-sm bg-yellow-200 text-inherit dark:bg-yellow-500/40">{{ segment.text }}</mark>
                        <span v-else>{{ segment.text }}</span>
                      </template>
                    </span>
                  </div>
                  <div v-if="matchingFeatures(product).length" class="space-y-1 text-xs text-muted-foreground">
                    <p v-for="match in matchingFeatures(product).slice(0, 3)" :key="match.index" class="line-clamp-2">
                      <span class="font-medium text-foreground">Matched feature · </span>
                      <template v-for="(segment, index) in highlight(match.feature)" :key="index">
                        <mark v-if="segment.matched" class="rounded-sm bg-yellow-200 text-inherit dark:bg-yellow-500/40">{{ segment.text }}</mark>
                        <span v-else>{{ segment.text }}</span>
                      </template>
                    </p>
                  </div>
                    </div>
                  </button>
                </ProductDetailDialog>
                <Button
                  type="button"
                  size="sm"
                  class="w-[6.25rem] shrink-0"
                  :variant="selectedIds.has(product.parent_asin) ? 'secondary' : 'outline'"
                  :disabled="modelValue.length >= 10 && !selectedIds.has(product.parent_asin)"
                  @click="selectedIds.has(product.parent_asin) ? removeById(product.parent_asin) : add(product)"
                >
                  <X v-if="selectedIds.has(product.parent_asin)" class="size-4" />
                  <Plus v-else class="size-4" />
                  {{ selectedIds.has(product.parent_asin) ? 'Remove' : 'Add' }}
                </Button>
              </div>
              </article>
            </div>

            <div v-if="error" class="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
              {{ error }}
            </div>
            <Button
              v-if="hasMore"
              type="button"
              variant="outline"
              class="w-full"
              :disabled="loadingMore"
              @click="loadMore"
            >
              <LoaderCircle v-if="loadingMore" class="size-4 animate-spin" />
              {{ loadingMore ? 'Loading…' : 'Load more products' }}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>

    <div v-if="modelValue.length" class="space-y-2">
      <div
        v-for="(product, index) in modelValue"
        :key="product.parent_asin"
        class="flex items-center gap-2 rounded-md border bg-card p-2"
      >
        <Badge variant="secondary" class="w-7 justify-center">{{ index + 1 }}</Badge>
        <ProductDetailDialog :product="product">
          <button type="button" class="flex min-w-0 flex-1 items-center gap-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring">
            <img
              v-if="product.thumb"
              :src="product.thumb"
              alt=""
              class="size-9 shrink-0 rounded-md border bg-white object-contain p-0.5"
              loading="lazy"
              decoding="async"
              referrerpolicy="no-referrer"
            />
            <div class="min-w-0 flex-1">
              <p class="truncate text-xs font-medium">{{ product.title }}</p>
              <p class="font-mono text-[11px] text-muted-foreground">{{ product.parent_asin }}</p>
            </div>
          </button>
        </ProductDetailDialog>
        <div class="flex">
          <Button type="button" size="icon-sm" variant="ghost" :disabled="index === 0" aria-label="Move up" @click="move(index, -1)">
            <ChevronUp class="size-3.5" />
          </Button>
          <Button type="button" size="icon-sm" variant="ghost" :disabled="index === modelValue.length - 1" aria-label="Move down" @click="move(index, 1)">
            <ChevronDown class="size-3.5" />
          </Button>
          <Button type="button" size="icon-sm" variant="ghost" aria-label="Remove" @click="remove(index)">
            <X class="size-3.5" />
          </Button>
        </div>
      </div>
    </div>
  </div>
</template>
