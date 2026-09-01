import type { ClassValue } from "clsx"
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function nonlinearScore(value: number): number {
  const score = Math.min(1, Math.max(0, value))
  if (score <= 0.1) return score * 2
  if (score <= 0.5) {
    const offset = score - 0.1
    return 0.2 + offset * 1.1 + offset * offset
  }
  if (score <= 0.6) return 0.8 + (score - 0.5)
  return 0.9 + (score - 0.6) * 0.25
}
