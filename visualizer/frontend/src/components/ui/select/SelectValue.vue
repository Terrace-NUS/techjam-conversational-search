<script setup lang="ts">
import type { SelectValueProps } from 'reka-ui'
import { SelectValue } from 'reka-ui'

const props = defineProps<SelectValueProps>()

let lastLabel = ''

function displayLabel(selectedLabel: string[], modelValue: unknown) {
  const isEmpty = modelValue == null
    || modelValue === ''
    || (Array.isArray(modelValue) && modelValue.length === 0)

  if (selectedLabel.length || isEmpty)
    lastLabel = selectedLabel.join(', ')

  return lastLabel || props.placeholder
}
</script>

<template>
  <SelectValue
    data-slot="select-value"
    v-bind="props"
    v-slot="{ selectedLabel, modelValue }"
  >
    <slot>{{ displayLabel(selectedLabel, modelValue) }}</slot>
  </SelectValue>
</template>
