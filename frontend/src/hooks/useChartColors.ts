"use client"

import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

/**
 * Hook to get chart colors from CSS variables (supports custom theme colors)
 * Returns primary, secondary, and background colors that respect the current theme
 */
export function useChartColors() {
  const { theme, resolvedTheme } = useTheme()
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