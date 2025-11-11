"use client"

import { oklchToHex, OklchColor } from "./theme-colors";

/**
 * Parse OKLCH CSS string to OKLCH object
 */
function parseOklch(oklchString: string): OklchColor | null {
  const match = oklchString.match(/oklch\(([\d.]+)\s+([\d.]+)\s+([\d.]+)\)/);
  if (!match) return null;
  return {
    l: parseFloat(match[1]),
    c: parseFloat(match[2]),
    h: parseFloat(match[3]),
  };
}

/**
 * Convert OKLCH CSS string to hex color
 */
export function oklchCssToHex(oklchString: string): string {
  const oklch = parseOklch(oklchString);
  if (!oklch) {
    // Fallback: try to use browser's computed color
    if (typeof window !== "undefined") {
      const div = document.createElement("div");
      div.style.color = oklchString;
      document.body.appendChild(div);
      const computed = window.getComputedStyle(div).color;
      document.body.removeChild(div);
      // Convert rgb() to hex
      const rgbMatch = computed.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
      if (rgbMatch) {
        const r = parseInt(rgbMatch[1]).toString(16).padStart(2, "0");
        const g = parseInt(rgbMatch[2]).toString(16).padStart(2, "0");
        const b = parseInt(rgbMatch[3]).toString(16).padStart(2, "0");
        return `#${r}${g}${b}`;
      }
    }
    return "#000000";
  }
  return oklchToHex(oklch);
}

/**
 * Get chart colors from CSS variables (supports custom theme colors)
 * Returns an array of 5 chart colors that can be used for data visualization
 */
export function getChartColors(): string[] {
  if (typeof window === 'undefined') {
    return [
      '#3b82f6', // chart-1
      '#10b981', // chart-2
      '#8b5cf6', // chart-3
      '#f59e0b', // chart-4
      '#ef4444', // chart-5
    ];
  }

  const computedStyle = getComputedStyle(document.documentElement);
  
  // Get chart colors from CSS variables (these are set by custom theme or defaults)
  const chart1 = computedStyle.getPropertyValue('--chart-1').trim();
  const chart2 = computedStyle.getPropertyValue('--chart-2').trim();
  const chart3 = computedStyle.getPropertyValue('--chart-3').trim();
  const chart4 = computedStyle.getPropertyValue('--chart-4').trim();
  const chart5 = computedStyle.getPropertyValue('--chart-5').trim();
  
  // Return chart colors (already in OKLCH format from CSS, or fallback to defaults)
  return [
    chart1 || 'oklch(0.646 0.222 41.116)',
    chart2 || 'oklch(0.6 0.118 184.704)',
    chart3 || 'oklch(0.398 0.07 227.392)',
    chart4 || 'oklch(0.828 0.189 84.429)',
    chart5 || 'oklch(0.769 0.188 70.08)',
  ];
}

/**
 * Get primary and background colors from CSS variables
 * @deprecated Use getChartColors() for chart colors, or access CSS variables directly
 */
export function getPrimaryAndBackground() {
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
  
  // Return colors (already in OKLCH format from CSS, or fallback)
  return {
    primary: primary || 'oklch(0.205 0 0)',
    background: background || 'oklch(1 0 0)'
  }
}