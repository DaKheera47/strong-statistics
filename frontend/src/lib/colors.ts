"use client"

export function getChartColors() {
  if (typeof window === 'undefined') {
    return {
      primary: '#0f172a',
      background: '#ffffff'
    }
  }

  const computedStyle = getComputedStyle(document.documentElement)
  
  // Get the actual color values from CSS variables
  const primary = computedStyle.getPropertyValue('--primary').trim()
  const background = computedStyle.getPropertyValue('--background').trim()
  
  // Convert OKLCH to hex if needed, or return fallback colors
  return {
    primary: primary ? `hsl(${primary})` : '#0f172a',
    background: background ? `hsl(${background})` : '#ffffff'
  }
}