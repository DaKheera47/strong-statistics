import { useState, useEffect } from 'react';
import { isLowerBetter } from '@/lib/exercise-config';

export interface ProgressiveOverloadDataPoint {
  date: string;
  maxWeight: number;
  reps: number;
  weekAgo?: number;
  monthAgo?: number;
  yearAgo?: number;
}

export interface ProgressiveOverloadData {
  exercise: string;
  data: ProgressiveOverloadDataPoint[];
}

function addDays(date: Date, days: number): Date {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

function findComparisonValue(data: Array<{date: string, maxWeight: number}>, targetDate: string): number | undefined {
  // Find the closest data point to the target date (within a reasonable range)
  const target = new Date(targetDate);
  const candidates = data.filter(d => {
    const dataDate = new Date(d.date);
    const daysDiff = Math.abs((dataDate.getTime() - target.getTime()) / (1000 * 60 * 60 * 24));
    return daysDiff <= 3; // Within 3 days tolerance
  });

  if (candidates.length === 0) return undefined;

  // Return the closest match
  return candidates.sort((a, b) => {
    const aDiff = Math.abs(new Date(a.date).getTime() - target.getTime());
    const bDiff = Math.abs(new Date(b.date).getTime() - target.getTime());
    return aDiff - bDiff;
  })[0].maxWeight;
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
        const response = await fetch('/api/max-weight-sparklines');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const rawData: Array<{exercise: string, date: string, maxWeight: number, reps: number}> = await response.json();
        
        // Filter for selected exercise
        const exerciseData = rawData
          .filter(row => row.exercise === selectedExercise)
          .sort((a, b) => a.date.localeCompare(b.date));

        if (exerciseData.length === 0) {
          setData({ exercise: selectedExercise, data: [] });
          setLoading(false);
          return;
        }

        // Create comparison data
        const processedData: ProgressiveOverloadDataPoint[] = exerciseData.map(point => {
          const currentDate = new Date(point.date);
          
          // Calculate comparison dates
          const weekAgoDate = addDays(currentDate, -7).toISOString().split('T')[0];
          const monthAgoDate = addDays(currentDate, -30).toISOString().split('T')[0];
          const yearAgoDate = addDays(currentDate, -365).toISOString().split('T')[0];

          // Find comparison values
          const weekAgo = findComparisonValue(exerciseData, weekAgoDate);
          const monthAgo = findComparisonValue(exerciseData, monthAgoDate);
          const yearAgo = findComparisonValue(exerciseData, yearAgoDate);

          return {
            date: point.date,
            maxWeight: point.maxWeight,
            reps: point.reps,
            weekAgo,
            monthAgo,
            yearAgo
          };
        });

        setData({
          exercise: selectedExercise,
          data: processedData
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