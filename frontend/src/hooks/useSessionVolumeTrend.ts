import { useState, useEffect } from 'react';

export interface SessionVolumeTrendData {
  date: string;
  volume: number;
  sets: number;
  duration_minutes: number | null;
  trendLine?: number; // For 7-day rolling average
}

export function useSessionVolumeTrend() {
  const [data, setData] = useState<SessionVolumeTrendData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/session-volume-trend');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const rawData: SessionVolumeTrendData[] = await response.json();
        
        // Calculate 7-day rolling average for trend line
        const dataWithTrend = rawData.map((item, index) => {
          const windowStart = Math.max(0, index - 6); // 7-day window including current day
          const window = rawData.slice(windowStart, index + 1);
          const trendLine = window.reduce((sum, d) => sum + d.volume, 0) / window.length;
          
          return {
            ...item,
            trendLine
          };
        });

        setData(dataWithTrend);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return { data, loading, error };
}