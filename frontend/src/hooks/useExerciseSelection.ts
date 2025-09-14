import { useState, useEffect } from "react";

export interface ExerciseWithLastActivity {
  name: string;
  lastActivityDate: string;
}

export function useExerciseSelection() {
  const [allExercises, setAllExercises] = useState<ExerciseWithLastActivity[]>(
    []
  );
  const [selectedExercises, setSelectedExercises] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchExercises() {
      try {
        setLoading(true);
        const response = await fetch("/api/recent-exercises");
        if (!response.ok) {
          throw new Error("Failed to fetch exercises");
        }
        const data: ExerciseWithLastActivity[] = await response.json();
        setAllExercises(data);

        // Select recent exercises by default (within last 2 weeks)
        const twoWeeksAgo = new Date();
        twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);
        const twoWeeksAgoStr = twoWeeksAgo.toISOString().split("T")[0];

        const recentExercises = data.filter(
          (ex) => ex.lastActivityDate >= twoWeeksAgoStr
        );

        // Ensure at least 2 exercises are selected
        const exercisesToSelect =
          recentExercises.length >= 2
            ? recentExercises
            : data.slice(0, Math.max(2, Math.min(data.length, 2)));

        setSelectedExercises(exercisesToSelect.map((ex) => ex.name));
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to fetch exercises"
        );
      } finally {
        setLoading(false);
      }
    }

    fetchExercises();
  }, []);

  // Wrapper to enforce minimum selection
  const setSelectedExercisesWithMinimum = (exercises: string[]) => {
    if (exercises.length >= 2) {
      setSelectedExercises(exercises);
    } else if (allExercises.length >= 2) {
      // If trying to set less than 2, keep current selection or set to first 2
      const fallback =
        selectedExercises.length >= 2
          ? selectedExercises
          : allExercises.slice(0, 2).map((ex) => ex.name);
      setSelectedExercises(fallback);
    }
  };

  return {
    allExercises,
    selectedExercises,
    setSelectedExercises: setSelectedExercisesWithMinimum,
    loading,
    error,
  };
}
