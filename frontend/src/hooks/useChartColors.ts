"use client"

import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

export function useChartColors() {
  const { theme, resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // Return default colors during SSR or before hydration
  if (!mounted) {
    return {
      primary: '#3b82f6',
      secondary: '#10b981', 
      background: '#ffffff'
    }
  }

  const isDark = resolvedTheme === 'dark'

  return {
    primary: isDark ? '#a855f7' : '#8b5cf6',
    secondary: isDark ? '#34d399' : '#10b981',
    background: isDark ? '#1f2937' : '#ffffff'
  }
}