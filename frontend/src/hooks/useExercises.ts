import { useState, useEffect } from 'react';

export interface Exercise {
  name: string;
}

export interface ExerciseWithLastActivity {
  name: string;
  lastActivityDate: string;
}

export function useExercises() {
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [allExercises, setAllExercises] = useState<ExerciseWithLastActivity[]>([]);
  const [selectedExercises, setSelectedExercises] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchExercises() {
      try {
        setLoading(true);
        const response = await fetch('/api/recent-exercises');
        if (!response.ok) {
          throw new Error('Failed to fetch exercises');
        }
        const data: ExerciseWithLastActivity[] = await response.json();
        setAllExercises(data);
        
        // Filter exercises active in the last 2 weeks
        const twoWeeksAgo = new Date();
        twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);
        const twoWeeksAgoStr = twoWeeksAgo.toISOString().split('T')[0];
        
        const recentExercises = data.filter(ex => 
          ex.lastActivityDate >= twoWeeksAgoStr
        );
        
        setExercises(recentExercises.map(ex => ({ name: ex.name })));
        
        // Select recent exercises by default
        setSelectedExercises(recentExercises.map(ex => ex.name));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch exercises');
      } finally {
        setLoading(false);
      }
    }

    fetchExercises();
  }, []);

  return {
    exercises,
    allExercises,
    selectedExercises,
    setSelectedExercises,
    loading,
    error
  };
}