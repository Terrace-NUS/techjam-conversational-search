<script setup lang="ts">
import type { DialogContentEmits, DialogContentProps } from 'reka-ui'

import type { HTMLAttributes } from 'vue'
import { XIcon } from '@lucide/vue'
import { reactiveOmit } from '@vueuse/core'
import { AnimatePresence, motion } from 'motion-v'
import {
  DialogClose,
  DialogContent,
  DialogPortal,
  useForwardPropsEmits,
} from 'reka-ui'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import DialogOverlay from './DialogOverlay.vue'

defineOptions({
  inheritAttrs: false,
})

const props = withDefaults(defineProps<DialogContentProps & { class?: HTMLAttributes['class'], showCloseButton?: boolean }>(), {
  showCloseButton: true,
})
const emits = defineEmits<DialogContentEmits>()

const delegatedProps = reactiveOmit(props, 'class')

const forwarded = useForwardPropsEmits(delegatedProps, emits)
</script>

<template>
  <DialogPortal>
    <DialogOverlay />
    <AnimatePresence>
      <DialogContent
        data-slot="dialog-content"
        v-bind="{ ...$attrs, ...forwarded }"
        as-child
      >
        <motion.div
          key="dialog-content"
          :initial="{ opacity: 0, y: 18, scale: 0.96 }"
          :animate="{ opacity: 1, y: 0, scale: 1 }"
          :exit="{ opacity: 0, y: 10, scale: 0.97 }"
          :transition="{ duration: 0.22, ease: 'easeOut' }"
          :class="cn('bg-popover text-popover-foreground ring-foreground/10 grid max-w-[calc(100%-2rem)] gap-4 rounded-xl p-4 text-sm ring-1 sm:max-w-sm fixed top-1/2 left-1/2 z-50 w-full -translate-x-1/2 -translate-y-1/2 outline-none', props.class)"
        >
          <slot />

          <DialogClose
            v-if="showCloseButton"
            data-slot="dialog-close"
            as-child
          >
            <Button variant="ghost" class="absolute top-2 right-2" size="icon-sm">
              <XIcon />
              <span class="sr-only">Close</span>
            </Button>
          </DialogClose>
        </motion.div>
      </DialogContent>
    </AnimatePresence>
  </DialogPortal>
</template>
