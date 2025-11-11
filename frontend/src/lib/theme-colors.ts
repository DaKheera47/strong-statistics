"use client"

/**
 * Theme color generation utilities using OKLCH color space
 * Generates offshoot colors (variants, complementary colors, etc.) from a primary color
 */

export interface OklchColor {
  l: number; // lightness: 0-1
  c: number; // chroma: 0-0.4
  h: number; // hue: 0-360
}

export interface ThemeColorPalette {
  primary: string;
  primaryForeground: string;
  secondary: string;
  secondaryForeground: string;
  accent: string;
  accentForeground: string;
  muted: string;
  mutedForeground: string;
  chart1: string;
  chart2: string;
  chart3: string;
  chart4: string;
  chart5: string;
  // Light mode colors
  light: {
    primary: string;
    primaryForeground: string;
    secondary: string;
    secondaryForeground: string;
    accent: string;
    accentForeground: string;
    muted: string;
    mutedForeground: string;
    chart1: string;
    chart2: string;
    chart3: string;
    chart4: string;
    chart5: string;
  };
  // Dark mode colors
  dark: {
    primary: string;
    primaryForeground: string;
    secondary: string;
    secondaryForeground: string;
    accent: string;
    accentForeground: string;
    muted: string;
    mutedForeground: string;
    chart1: string;
    chart2: string;
    chart3: string;
    chart4: string;
    chart5: string;
  };
}

/**
 * Convert hex color to RGB
 */
function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  return [
    parseInt(result[1], 16),
    parseInt(result[2], 16),
    parseInt(result[3], 16),
  ];
}

/**
 * Convert RGB to linear RGB (0-1 range)
 */
function rgbToLinear(rgb: [number, number, number]): [number, number, number] {
  return rgb.map((val) => {
    val = val / 255;
    return val <= 0.04045 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
  }) as [number, number, number];
}

/**
 * Convert linear RGB to OKLab
 */
function linearRgbToOklab(rgb: [number, number, number]): [number, number, number] {
  const [r, g, b] = rgb;
  const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  return [
    0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  ];
}

/**
 * Convert OKLab to OKLCH
 */
function oklabToOklch(lab: [number, number, number]): OklchColor {
  const [l, a, b] = lab;
  const c = Math.sqrt(a * a + b * b);
  let h = Math.atan2(b, a) * (180 / Math.PI);
  if (h < 0) h += 360;
  return { l, c, h };
}

/**
 * Convert hex color to OKLCH
 */
export function hexToOklch(hex: string): OklchColor {
  const rgb = hexToRgb(hex);
  const linearRgb = rgbToLinear(rgb);
  const oklab = linearRgbToOklab(linearRgb);
  return oklabToOklch(oklab);
}

/**
 * Convert OKLCH to OKLab
 */
function oklchToOklab(oklch: OklchColor): [number, number, number] {
  const { l, c, h } = oklch;
  const hRad = (h * Math.PI) / 180;
  const a = c * Math.cos(hRad);
  const b = c * Math.sin(hRad);
  return [l, a, b];
}

/**
 * Convert OKLab to linear RGB
 */
function oklabToLinearRgb(lab: [number, number, number]): [number, number, number] {
  const [l, a, b] = lab;
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = l - 0.0894841775 * a - 1.291485548 * b;

  const lLinear = l_ * l_ * l_;
  const mLinear = m_ * m_ * m_;
  const sLinear = s_ * s_ * s_;

  return [
    +4.0767416621 * lLinear - 3.3077115913 * mLinear + 0.2309699292 * sLinear,
    -1.2684380046 * lLinear + 2.6097574011 * mLinear - 0.3413193965 * sLinear,
    -0.0041960863 * lLinear - 0.7034186147 * mLinear + 1.707614701 * sLinear,
  ];
}

/**
 * Convert linear RGB to RGB
 */
function linearRgbToRgb(rgb: [number, number, number]): [number, number, number] {
  return rgb.map((val) => {
    val = val <= 0.0031308 ? 12.92 * val : 1.055 * Math.pow(val, 1 / 2.4) - 0.055;
    return Math.round(Math.max(0, Math.min(255, val * 255)));
  }) as [number, number, number];
}

/**
 * Convert OKLCH to hex color
 */
export function oklchToHex(oklch: OklchColor): string {
  const oklab = oklchToOklab(oklch);
  const linearRgb = oklabToLinearRgb(oklab);
  const rgb = linearRgbToRgb(linearRgb);
  return `#${rgb.map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

/**
 * Format OKLCH color as CSS oklch() string
 */
export function oklchToCss(oklch: OklchColor): string {
  return `oklch(${oklch.l.toFixed(3)} ${oklch.c.toFixed(3)} ${oklch.h.toFixed(1)})`;
}

/**
 * Calculate relative luminance for contrast checking
 */
function getLuminance(oklch: OklchColor): number {
  return oklch.l;
}

/**
 * Calculate contrast ratio between two colors
 */
function getContrastRatio(color1: OklchColor, color2: OklchColor): number {
  const l1 = getLuminance(color1);
  const l2 = getLuminance(color2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Generate a foreground color with sufficient contrast
 */
function generateForegroundColor(
  background: OklchColor,
  minContrast: number = 4.5
): OklchColor {
  // Try white first
  const white: OklchColor = { l: 0.985, c: 0, h: 0 };
  if (getContrastRatio(background, white) >= minContrast) {
    return white;
  }

  // Try black
  const black: OklchColor = { l: 0.145, c: 0, h: 0 };
  if (getContrastRatio(background, black) >= minContrast) {
    return black;
  }

  // Adjust lightness to achieve contrast
  const bgL = background.l;
  if (bgL > 0.5) {
    // Light background, use dark foreground
    return { l: Math.max(0.145, bgL - 0.4), c: 0, h: 0 };
  } else {
    // Dark background, use light foreground
    return { l: Math.min(0.985, bgL + 0.4), c: 0, h: 0 };
  }
}

/**
 * Generate chart colors with varying hues around the color wheel
 */
function generateChartColors(primary: OklchColor): string[] {
  const { h } = primary;
  const colors: string[] = [];

  // Generate 5 colors with hues spaced around the color wheel
  // Use complementary and analogous colors
  const hueOffsets = [
    0, // Primary color
    60, // Analogous
    120, // Triadic
    180, // Complementary
    240, // Split complementary
  ];

  hueOffsets.forEach((offset, index) => {
    const newHue = (h + offset) % 360;
    // Vary chroma and lightness for visual distinction (deterministic based on index)
    // Use sine/cosine functions to create smooth variations
    const chromaVariation = 0.7 + (Math.sin(index * 1.5) * 0.15 + 0.15);
    const lightnessVariation = 0.5 + (Math.cos(index * 1.2) * 0.1);
    const chroma = primary.c * chromaVariation;
    const lightness = lightnessVariation;
    const chartColor: OklchColor = {
      l: Math.max(0.3, Math.min(0.85, lightness)),
      c: Math.max(0.08, Math.min(0.3, chroma)),
      h: newHue,
    };
    colors.push(oklchToCss(chartColor));
  });

  return colors;
}

/**
 * Generate a complete theme color palette from a primary color
 */
export function generateThemePalette(primaryHex: string): ThemeColorPalette {
  const primaryOklch = hexToOklch(primaryHex);

  // Light mode colors
  const lightPrimary: OklchColor = {
    l: Math.max(0.2, Math.min(0.3, primaryOklch.l - 0.1)),
    c: primaryOklch.c,
    h: primaryOklch.h,
  };
  const lightPrimaryForeground = generateForegroundColor(lightPrimary);

  const lightSecondary: OklchColor = {
    l: 0.97,
    c: 0,
    h: primaryOklch.h,
  };
  const lightSecondaryForeground: OklchColor = {
    l: 0.205,
    c: 0,
    h: 0,
  };

  const lightAccent: OklchColor = {
    l: 0.97,
    c: primaryOklch.c * 0.3,
    h: primaryOklch.h,
  };
  const lightAccentForeground: OklchColor = {
    l: 0.205,
    c: 0,
    h: 0,
  };

  const lightMuted: OklchColor = {
    l: 0.97,
    c: 0,
    h: primaryOklch.h,
  };
  const lightMutedForeground: OklchColor = {
    l: 0.556,
    c: 0,
    h: 0,
  };

  // Dark mode colors
  const darkPrimary: OklchColor = {
    l: Math.min(0.95, Math.max(0.85, primaryOklch.l + 0.15)),
    c: primaryOklch.c,
    h: primaryOklch.h,
  };
  const darkPrimaryForeground = generateForegroundColor(darkPrimary);

  const darkSecondary: OklchColor = {
    l: 0.269,
    c: 0,
    h: primaryOklch.h,
  };
  const darkSecondaryForeground: OklchColor = {
    l: 0.985,
    c: 0,
    h: 0,
  };

  const darkAccent: OklchColor = {
    l: 0.269,
    c: primaryOklch.c * 0.3,
    h: primaryOklch.h,
  };
  const darkAccentForeground: OklchColor = {
    l: 0.985,
    c: 0,
    h: 0,
  };

  const darkMuted: OklchColor = {
    l: 0.269,
    c: 0,
    h: primaryOklch.h,
  };
  const darkMutedForeground: OklchColor = {
    l: 0.708,
    c: 0,
    h: 0,
  };

  // Generate chart colors for both modes
  const lightChartColors = generateChartColors(lightPrimary);
  const darkChartColors = generateChartColors(darkPrimary);

  return {
    primary: oklchToCss(lightPrimary),
    primaryForeground: oklchToCss(lightPrimaryForeground),
    secondary: oklchToCss(lightSecondary),
    secondaryForeground: oklchToCss(lightSecondaryForeground),
    accent: oklchToCss(lightAccent),
    accentForeground: oklchToCss(lightAccentForeground),
    muted: oklchToCss(lightMuted),
    mutedForeground: oklchToCss(lightMutedForeground),
    chart1: lightChartColors[0],
    chart2: lightChartColors[1],
    chart3: lightChartColors[2],
    chart4: lightChartColors[3],
    chart5: lightChartColors[4],
    light: {
      primary: oklchToCss(lightPrimary),
      primaryForeground: oklchToCss(lightPrimaryForeground),
      secondary: oklchToCss(lightSecondary),
      secondaryForeground: oklchToCss(lightSecondaryForeground),
      accent: oklchToCss(lightAccent),
      accentForeground: oklchToCss(lightAccentForeground),
      muted: oklchToCss(lightMuted),
      mutedForeground: oklchToCss(lightMutedForeground),
      chart1: lightChartColors[0],
      chart2: lightChartColors[1],
      chart3: lightChartColors[2],
      chart4: lightChartColors[3],
      chart5: lightChartColors[4],
    },
    dark: {
      primary: oklchToCss(darkPrimary),
      primaryForeground: oklchToCss(darkPrimaryForeground),
      secondary: oklchToCss(darkSecondary),
      secondaryForeground: oklchToCss(darkSecondaryForeground),
      accent: oklchToCss(darkAccent),
      accentForeground: oklchToCss(darkAccentForeground),
      muted: oklchToCss(darkMuted),
      mutedForeground: oklchToCss(darkMutedForeground),
      chart1: darkChartColors[0],
      chart2: darkChartColors[1],
      chart3: darkChartColors[2],
      chart4: darkChartColors[3],
      chart5: darkChartColors[4],
    },
  };
}

