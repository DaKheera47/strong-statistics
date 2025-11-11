"use client"

import { useState, useEffect, useCallback } from "react";
import { useTheme } from "next-themes";
import {
  getPrimaryColor,
  setPrimaryColor,
  getThemeColors,
  setThemeColors,
} from "@/lib/localStorage";
import {
  generateThemePalette,
  ThemeColorPalette,
} from "@/lib/theme-colors";

/**
 * Hook to manage customizable theme colors
 * Handles loading from localStorage, generating color palettes, and applying to CSS variables
 */
export function useThemeColors() {
  const { theme, resolvedTheme } = useTheme();
  const [primaryColor, setPrimaryColorState] = useState<string | null>(null);
  const [colorPalette, setColorPaletteState] = useState<ThemeColorPalette | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load colors from localStorage on mount
  useEffect(() => {
    const savedPrimary = getPrimaryColor();
    const savedPalette = getThemeColors();

    if (savedPrimary && savedPalette) {
      setPrimaryColorState(savedPrimary);
      // Reconstruct palette object from saved flat structure
      const palette: ThemeColorPalette = {
        primary: savedPalette.primary || "",
        primaryForeground: savedPalette.primaryForeground || "",
        secondary: savedPalette.secondary || "",
        secondaryForeground: savedPalette.secondaryForeground || "",
        accent: savedPalette.accent || "",
        accentForeground: savedPalette.accentForeground || "",
        muted: savedPalette.muted || "",
        mutedForeground: savedPalette.mutedForeground || "",
        chart1: savedPalette.chart1 || "",
        chart2: savedPalette.chart2 || "",
        chart3: savedPalette.chart3 || "",
        chart4: savedPalette.chart4 || "",
        chart5: savedPalette.chart5 || "",
        light: {
          primary: savedPalette["light.primary"] || "",
          primaryForeground: savedPalette["light.primaryForeground"] || "",
          secondary: savedPalette["light.secondary"] || "",
          secondaryForeground: savedPalette["light.secondaryForeground"] || "",
          accent: savedPalette["light.accent"] || "",
          accentForeground: savedPalette["light.accentForeground"] || "",
          muted: savedPalette["light.muted"] || "",
          mutedForeground: savedPalette["light.mutedForeground"] || "",
          chart1: savedPalette["light.chart1"] || "",
          chart2: savedPalette["light.chart2"] || "",
          chart3: savedPalette["light.chart3"] || "",
          chart4: savedPalette["light.chart4"] || "",
          chart5: savedPalette["light.chart5"] || "",
        },
        dark: {
          primary: savedPalette["dark.primary"] || "",
          primaryForeground: savedPalette["dark.primaryForeground"] || "",
          secondary: savedPalette["dark.secondary"] || "",
          secondaryForeground: savedPalette["dark.secondaryForeground"] || "",
          accent: savedPalette["dark.accent"] || "",
          accentForeground: savedPalette["dark.accentForeground"] || "",
          muted: savedPalette["dark.muted"] || "",
          mutedForeground: savedPalette["dark.mutedForeground"] || "",
          chart1: savedPalette["dark.chart1"] || "",
          chart2: savedPalette["dark.chart2"] || "",
          chart3: savedPalette["dark.chart3"] || "",
          chart4: savedPalette["dark.chart4"] || "",
          chart5: savedPalette["dark.chart5"] || "",
        },
      };
      setColorPaletteState(palette);
    }
    setIsLoaded(true);
  }, []);

  // Apply colors to CSS variables
  const applyColorsToCSS = useCallback(
    (palette: ThemeColorPalette | null) => {
      if (typeof window === "undefined" || !palette) {
        return;
      }

      const root = document.documentElement;
      const isDark = resolvedTheme === "dark";
      const modePalette = isDark ? palette.dark : palette.light;

      // Apply light mode colors (base)
      root.style.setProperty("--primary", palette.light.primary);
      root.style.setProperty("--primary-foreground", palette.light.primaryForeground);
      root.style.setProperty("--secondary", palette.light.secondary);
      root.style.setProperty("--secondary-foreground", palette.light.secondaryForeground);
      root.style.setProperty("--accent", palette.light.accent);
      root.style.setProperty("--accent-foreground", palette.light.accentForeground);
      root.style.setProperty("--muted", palette.light.muted);
      root.style.setProperty("--muted-foreground", palette.light.mutedForeground);
      root.style.setProperty("--chart-1", palette.light.chart1);
      root.style.setProperty("--chart-2", palette.light.chart2);
      root.style.setProperty("--chart-3", palette.light.chart3);
      root.style.setProperty("--chart-4", palette.light.chart4);
      root.style.setProperty("--chart-5", palette.light.chart5);

      // Apply dark mode colors if dark mode is active
      if (isDark) {
        root.style.setProperty("--primary", modePalette.primary);
        root.style.setProperty("--primary-foreground", modePalette.primaryForeground);
        root.style.setProperty("--secondary", modePalette.secondary);
        root.style.setProperty("--secondary-foreground", modePalette.secondaryForeground);
        root.style.setProperty("--accent", modePalette.accent);
        root.style.setProperty("--accent-foreground", modePalette.accentForeground);
        root.style.setProperty("--muted", modePalette.muted);
        root.style.setProperty("--muted-foreground", modePalette.mutedForeground);
        root.style.setProperty("--chart-1", modePalette.chart1);
        root.style.setProperty("--chart-2", modePalette.chart2);
        root.style.setProperty("--chart-3", modePalette.chart3);
        root.style.setProperty("--chart-4", modePalette.chart4);
        root.style.setProperty("--chart-5", modePalette.chart5);
      }
    },
    [resolvedTheme]
  );

  // Apply colors when palette or theme changes
  useEffect(() => {
    if (isLoaded && colorPalette) {
      applyColorsToCSS(colorPalette);
    }
  }, [colorPalette, resolvedTheme, isLoaded, applyColorsToCSS]);

  // Update primary color and regenerate palette
  const updatePrimaryColor = useCallback((hexColor: string) => {
    try {
      const palette = generateThemePalette(hexColor);
      setPrimaryColorState(hexColor);
      setColorPaletteState(palette);

      // Flatten palette for storage
      const flatPalette: Record<string, string> = {
        primary: palette.primary,
        primaryForeground: palette.primaryForeground,
        secondary: palette.secondary,
        secondaryForeground: palette.secondaryForeground,
        accent: palette.accent,
        accentForeground: palette.accentForeground,
        muted: palette.muted,
        mutedForeground: palette.mutedForeground,
        chart1: palette.chart1,
        chart2: palette.chart2,
        chart3: palette.chart3,
        chart4: palette.chart4,
        chart5: palette.chart5,
        "light.primary": palette.light.primary,
        "light.primaryForeground": palette.light.primaryForeground,
        "light.secondary": palette.light.secondary,
        "light.secondaryForeground": palette.light.secondaryForeground,
        "light.accent": palette.light.accent,
        "light.accentForeground": palette.light.accentForeground,
        "light.muted": palette.light.muted,
        "light.mutedForeground": palette.light.mutedForeground,
        "light.chart1": palette.light.chart1,
        "light.chart2": palette.light.chart2,
        "light.chart3": palette.light.chart3,
        "light.chart4": palette.light.chart4,
        "light.chart5": palette.light.chart5,
        "dark.primary": palette.dark.primary,
        "dark.primaryForeground": palette.dark.primaryForeground,
        "dark.secondary": palette.dark.secondary,
        "dark.secondaryForeground": palette.dark.secondaryForeground,
        "dark.accent": palette.dark.accent,
        "dark.accentForeground": palette.dark.accentForeground,
        "dark.muted": palette.dark.muted,
        "dark.mutedForeground": palette.dark.mutedForeground,
        "dark.chart1": palette.dark.chart1,
        "dark.chart2": palette.dark.chart2,
        "dark.chart3": palette.dark.chart3,
        "dark.chart4": palette.dark.chart4,
        "dark.chart5": palette.dark.chart5,
      };

      setPrimaryColor(hexColor);
      setThemeColors(flatPalette);
    } catch (error) {
      console.error("Failed to update primary color:", error);
    }
  }, []);

  // Reset to default colors
  const resetColors = useCallback(() => {
    setPrimaryColorState(null);
    setColorPaletteState(null);
    setPrimaryColor(null);
    setThemeColors(null);

    // Remove CSS variables to fall back to defaults
    if (typeof window !== "undefined") {
      const root = document.documentElement;
      root.style.removeProperty("--primary");
      root.style.removeProperty("--primary-foreground");
      root.style.removeProperty("--secondary");
      root.style.removeProperty("--secondary-foreground");
      root.style.removeProperty("--accent");
      root.style.removeProperty("--accent-foreground");
      root.style.removeProperty("--muted");
      root.style.removeProperty("--muted-foreground");
      root.style.removeProperty("--chart-1");
      root.style.removeProperty("--chart-2");
      root.style.removeProperty("--chart-3");
      root.style.removeProperty("--chart-4");
      root.style.removeProperty("--chart-5");
    }
  }, []);

  return {
    primaryColor,
    colorPalette,
    isLoaded,
    updatePrimaryColor,
    resetColors,
  };
}

