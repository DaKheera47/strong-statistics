import { useState, useEffect } from 'react';

export interface RepRangeDistributionData {
  exercise: string;
  date: string;
  range_1_5: number;
  range_6_12: number;
  range_13_20: number;
  range_20_plus: number;
}

export interface ExerciseRepRangeData {
  exercise: string;
  data: Array<{
    date: string;
    range_1_5: number;
    range_6_12: number;
    range_13_20: number;
    range_20_plus: number;
    total: number;
  }>;
}

export function useRepRangeDistribution(selectedExercises: string[] = []) {
  const [data, setData] = useState<ExerciseRepRangeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/rep-range-distribution');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const rawData: RepRangeDistributionData[] = await response.json();
        
        // Group data by exercise
        const grouped = rawData.reduce((acc, row) => {
          if (!acc[row.exercise]) {
            acc[row.exercise] = [];
          }
          acc[row.exercise].push({
            date: row.date,
            range_1_5: row.range_1_5,
            range_6_12: row.range_6_12,
            range_13_20: row.range_13_20,
            range_20_plus: row.range_20_plus,
            total: row.range_1_5 + row.range_6_12 + row.range_13_20 + row.range_20_plus
          });
          return acc;
        }, {} as Record<string, Array<{date: string, range_1_5: number, range_6_12: number, range_13_20: number, range_20_plus: number, total: number}>>);

        // Transform to final format with filtering
        const transformed = Object.entries(grouped)
          .filter(([exercise]) => selectedExercises.length === 0 || selectedExercises.includes(exercise))
          .map(([exercise, exerciseData]) => {
            // Sort by date to ensure proper ordering
            const sortedData = exerciseData.sort((a, b) => a.date.localeCompare(b.date));
            
            return {
              exercise,
              data: sortedData
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