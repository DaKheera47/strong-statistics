"use client"

import { useTheme } from 'next-themes'
import { useEffect, useState, useMemo } from 'react'
import { getChartColors } from '@/lib/colors'

/**
 * Hook to get chart colors from CSS variables (supports custom theme colors)
 * Returns primary, secondary, and background colors that respect the current theme
 */
export function useChartColors() {
  const { resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  const [colors, setColors] = useState({
    primary: '#3b82f6',
    secondary: '#10b981', 
    background: '#ffffff'
  })

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return

    const computedStyle = getComputedStyle(document.documentElement)
    
    // Get colors from CSS variables (these respect custom theme colors)
    const primary = computedStyle.getPropertyValue('--primary').trim()
    const secondary = computedStyle.getPropertyValue('--secondary').trim()
    const background = computedStyle.getPropertyValue('--background').trim()
    
    setColors({
      primary: primary || '#3b82f6',
      secondary: secondary || '#10b981',
      background: background || '#ffffff'
    })
  }, [mounted, resolvedTheme])

  // Return default colors during SSR or before hydration
  if (!mounted) {
    return {
      primary: '#3b82f6',
      secondary: '#10b981', 
      background: '#ffffff'
    }
  }

  return colors
}

/**
 * Hook to get chart colors array from CSS variables (memoized)
 * Returns an array of 5 chart colors that can be used for data visualization
 * This hook memoizes the result and only re-reads when theme changes
 */
export function useChartColorsArray(): string[] {
  const { resolvedTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return useMemo(() => {
    if (!mounted) {
      // SSR fallback
      return [
        '#3b82f6', // chart-1
        '#10b981', // chart-2
        '#8b5cf6', // chart-3
        '#f59e0b', // chart-4
        '#ef4444', // chart-5
      ]
    }
    return getChartColors()
  }, [mounted, resolvedTheme])
}