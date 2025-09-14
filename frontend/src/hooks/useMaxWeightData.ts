import { useState, useEffect } from 'react';
import { isLowerBetter } from '@/lib/exercise-config';

export interface MaxWeightSparklineData {
  exercise: string;
  date: string;
  maxWeight: number;
  reps: number | null;
  distance: number | null;
}

export interface ExerciseMaxWeightData {
  exercise: string;
  data: Array<{
    date: string;
    maxWeight: number;
    reps: number | null;
    distance: number | null;
  }>;
  latestValue: number;
  delta: number;
}

export function useMaxWeightData(selectedExercises: string[] = []) {
  const [data, setData] = useState<ExerciseMaxWeightData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/max-weight-sparklines');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const rawData: MaxWeightSparklineData[] = await response.json();
        
        // Group data by exercise
        const grouped = rawData.reduce((acc, row) => {
          if (!acc[row.exercise]) {
            acc[row.exercise] = [];
          }
          acc[row.exercise].push({
            date: row.date,
            maxWeight: row.maxWeight,
            reps: row.reps,
            distance: row.distance
          });
          return acc;
        }, {} as Record<string, Array<{date: string, maxWeight: number, reps: number | null, distance: number | null}>>);

        // Transform to final format with calculations
        const transformed = Object.entries(grouped)
          .filter(([exercise]) => selectedExercises.length === 0 || selectedExercises.includes(exercise))
          .map(([exercise, exerciseData]) => {
            // Sort by date to ensure proper ordering
            const sortedData = exerciseData.sort((a, b) => a.date.localeCompare(b.date));
            const latestValue = sortedData[sortedData.length - 1]?.maxWeight || 0;
            const previousValue = sortedData[sortedData.length - 2]?.maxWeight || 0;
            
            // Calculate delta based on exercise type
            let delta = 0;
            if (previousValue > 0) {
              if (isLowerBetter(exercise)) {
                // For lower-is-better exercises, improvement is when current < previous
                delta = ((previousValue - latestValue) / previousValue) * 100;
              } else {
                // For normal exercises, improvement is when current > previous  
                delta = ((latestValue - previousValue) / previousValue) * 100;
              }
            }

            return {
              exercise,
              data: sortedData,
              latestValue,
              delta
            };
          });

        setData(transformed);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [selectedExercises]);

  return { data, loading, error };
}