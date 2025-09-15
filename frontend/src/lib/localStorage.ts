/**
 * Utilities for handling localStorage operations with TypeScript support
 */

// Prefix for all keys to avoid conflicts
const KEY_PREFIX = 'strong-statistics-';

// Configuration keys
export const STORAGE_KEYS = {
  VOLUME_SPARKLINES_LIMIT: `${KEY_PREFIX}volume-sparklines-limit`,
  MAX_WEIGHT_SPARKLINES_LIMIT: `${KEY_PREFIX}max-weight-sparklines-limit`,
  REP_RANGE_DISTRIBUTION_LIMIT: `${KEY_PREFIX}rep-range-distribution-limit`,
} as const;

// Default limits
export const DEFAULT_LIMITS = {
  VOLUME_SPARKLINES: 12,
  MAX_WEIGHT_SPARKLINES: 12,
  REP_RANGE_DISTRIBUTION: 9,
} as const;

/**
 * Safely get a value from localStorage
 */
export function getFromStorage<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') {
    return fallback;
  }

  try {
    const item = localStorage.getItem(key);
    if (item === null) {
      return fallback;
    }
    return JSON.parse(item) as T;
  } catch (error) {
    console.warn(`Failed to parse localStorage item '${key}':`, error);
    return fallback;
  }
}

/**
 * Safely set a value in localStorage
 */
export function setInStorage<T>(key: string, value: T): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`Failed to save to localStorage '${key}':`, error);
  }
}

/**
 * Remove a value from localStorage
 */
export function removeFromStorage(key: string): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn(`Failed to remove from localStorage '${key}':`, error);
  }
}

/**
 * Get display limit for volume sparklines
 */
export function getVolumeSparklineLimit(): number {
  return getFromStorage(STORAGE_KEYS.VOLUME_SPARKLINES_LIMIT, DEFAULT_LIMITS.VOLUME_SPARKLINES);
}

/**
 * Set display limit for volume sparklines
 */
export function setVolumeSparklineLimit(limit: number): void {
  setInStorage(STORAGE_KEYS.VOLUME_SPARKLINES_LIMIT, limit);
}

/**
 * Get display limit for max weight sparklines
 */
export function getMaxWeightSparklineLimit(): number {
  return getFromStorage(STORAGE_KEYS.MAX_WEIGHT_SPARKLINES_LIMIT, DEFAULT_LIMITS.MAX_WEIGHT_SPARKLINES);
}

/**
 * Set display limit for max weight sparklines
 */
export function setMaxWeightSparklineLimit(limit: number): void {
  setInStorage(STORAGE_KEYS.MAX_WEIGHT_SPARKLINES_LIMIT, limit);
}

/**
 * Get display limit for rep range distributions
 */
export function getRepRangeDistributionLimit(): number {
  return getFromStorage(STORAGE_KEYS.REP_RANGE_DISTRIBUTION_LIMIT, DEFAULT_LIMITS.REP_RANGE_DISTRIBUTION);
}

/**
 * Set display limit for rep range distributions
 */
export function setRepRangeDistributionLimit(limit: number): void {
  setInStorage(STORAGE_KEYS.REP_RANGE_DISTRIBUTION_LIMIT, limit);
}