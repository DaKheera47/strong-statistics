"use client"

import { useEffect } from "react";
import { useThemeColors } from "@/hooks/useThemeColors";

/**
 * Provider component that applies custom theme colors from localStorage
 * to CSS variables. Works alongside ThemeProvider for dark/light mode.
 */
export function DynamicThemeProvider({ children }: { children: React.ReactNode }) {
  const { isLoaded } = useThemeColors();

  // Ensure colors are applied before rendering children
  useEffect(() => {
    // Colors are applied automatically by useThemeColors hook
    // This component just ensures the hook is initialized
  }, []);

  return <>{children}</>;
}

