import { useState, useEffect } from 'react';

export interface ProgressiveOverloadDataPoint {
  date: string;
  maxWeight: number;
  reps: number;
  weekAgo?: number;
  monthAgo?: number;
  yearAgo?: number;
}

export interface ProgressiveVolumeDataPoint {
  date: string;
  volume: number;
  sets: number;
  weekAgo?: number;
  monthAgo?: number;
  yearAgo?: number;
}

export interface ProgressiveOverloadData {
  exercise: string;
  maxWeight: ProgressiveOverloadDataPoint[];
  volume: ProgressiveVolumeDataPoint[];
}

function addDays(date: Date, days: number): Date {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

function createComparisonPoint<T extends { date: string }>(
  data: T[],
  targetDate: string,
  valueSelector: (point: T) => number
) {
  const target = new Date(targetDate);
  const candidates = data.filter(d => {
    const dataDate = new Date(d.date);
    const daysDiff = Math.abs((dataDate.getTime() - target.getTime()) / (1000 * 60 * 60 * 24));
    return daysDiff <= 3;
  });

  if (candidates.length === 0) return undefined;

  return candidates
    .sort((a, b) => {
      const aDiff = Math.abs(new Date(a.date).getTime() - target.getTime());
      const bDiff = Math.abs(new Date(b.date).getTime() - target.getTime());
      return aDiff - bDiff;
    })
    .map(valueSelector)[0];
}

export function useProgressiveOverloadData(selectedExercise: string | null) {
  const [data, setData] = useState<ProgressiveOverloadData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      if (!selectedExercise) {
        setData(null);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);
        const [weightResponse, volumeResponse] = await Promise.all([
          fetch('/api/max-weight-sparklines'),
          fetch('/api/volume-sparklines'),
        ]);

        if (!weightResponse.ok) {
          throw new Error('Failed to fetch max weight data');
        }
        if (!volumeResponse.ok) {
          throw new Error('Failed to fetch volume data');
        }

        const rawWeightData: Array<{exercise: string, date: string, maxWeight: number, reps: number}> = await weightResponse.json();
        const rawVolumeData: Array<{exercise: string, date: string, volume: number, sets: number}> = await volumeResponse.json();

        const exerciseWeightData = rawWeightData
          .filter(row => row.exercise === selectedExercise)
          .sort((a, b) => a.date.localeCompare(b.date));

        const exerciseVolumeData = rawVolumeData
          .filter(row => row.exercise === selectedExercise)
          .sort((a, b) => a.date.localeCompare(b.date));

        if (exerciseWeightData.length === 0 && exerciseVolumeData.length === 0) {
          setData({ exercise: selectedExercise, maxWeight: [], volume: [] });
          setLoading(false);
          return;
        }

        const processedWeightData: ProgressiveOverloadDataPoint[] = exerciseWeightData.map(point => {
          const currentDate = new Date(point.date);
          
          // Calculate comparison dates
          const weekAgoDate = addDays(currentDate, -7).toISOString().split('T')[0];
          const monthAgoDate = addDays(currentDate, -30).toISOString().split('T')[0];
          const yearAgoDate = addDays(currentDate, -365).toISOString().split('T')[0];

          // Find comparison values
          const weekAgo = createComparisonPoint(exerciseWeightData, weekAgoDate, d => d.maxWeight);
          const monthAgo = createComparisonPoint(exerciseWeightData, monthAgoDate, d => d.maxWeight);
          const yearAgo = createComparisonPoint(exerciseWeightData, yearAgoDate, d => d.maxWeight);

          return {
            date: point.date,
            maxWeight: point.maxWeight,
            reps: point.reps,
            weekAgo,
            monthAgo,
            yearAgo
          };
        });

        const processedVolumeData: ProgressiveVolumeDataPoint[] = exerciseVolumeData.map(point => {
          const currentDate = new Date(point.date);

          const weekAgoDate = addDays(currentDate, -7).toISOString().split('T')[0];
          const monthAgoDate = addDays(currentDate, -30).toISOString().split('T')[0];
          const yearAgoDate = addDays(currentDate, -365).toISOString().split('T')[0];

          const weekAgo = createComparisonPoint(exerciseVolumeData, weekAgoDate, d => d.volume);
          const monthAgo = createComparisonPoint(exerciseVolumeData, monthAgoDate, d => d.volume);
          const yearAgo = createComparisonPoint(exerciseVolumeData, yearAgoDate, d => d.volume);

          return {
            date: point.date,
            volume: point.volume,
            sets: point.sets,
            weekAgo,
            monthAgo,
            yearAgo,
          };
        });

        setData({
          exercise: selectedExercise,
          maxWeight: processedWeightData,
          volume: processedVolumeData,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [selectedExercise]);

  return { data, loading, error };
}
