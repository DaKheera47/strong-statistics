export interface ExerciseConfig {
  name: string;
  lowerIsBetter?: boolean;
  displayDistanceInsteadOfReps?: boolean;
  distanceUnit?: string;
}

export const EXERCISE_CONFIGS: Record<string, ExerciseConfig> = {
  'Pull Up (Assisted)': {
    name: 'Pull Up (Assisted)',
    lowerIsBetter: true,
  },
  'Kettlebell Carry': {
    name: 'Kettlebell Carry',
    displayDistanceInsteadOfReps: true,
    distanceUnit: 'm',
  },
};

export function getExerciseConfig(exerciseName: string): ExerciseConfig {
  return EXERCISE_CONFIGS[exerciseName] || { name: exerciseName };
}

export function isLowerBetter(exerciseName: string): boolean {
  return getExerciseConfig(exerciseName).lowerIsBetter || false;
}

export function shouldDisplayDistance(exerciseName: string): boolean {
  return getExerciseConfig(exerciseName).displayDistanceInsteadOfReps || false;
}

export function getDistanceUnit(exerciseName: string): string {
  return getExerciseConfig(exerciseName).distanceUnit || 'm';
}