"use client"

import { useThemeColors } from "@/hooks/useThemeColors";

/**
 * Provider component that applies custom theme colors from localStorage
 * to CSS variables. Works alongside ThemeProvider for dark/light mode.
 */
export function DynamicThemeProvider({ children }: { children: React.ReactNode }) {
  // Initialize the hook to ensure colors are applied
  useThemeColors();

  return <>{children}</>;
}

