"use client";

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { cn } from "@/lib/utils";
import { SparklineCard } from "./SparklineCard";

export interface WorkoutDetailData {
  date: string;
  workout_name: string;
  duration_minutes: number | null;
  exercises: {
    exercise: string;
    sets: {
      set_order: number;
      weight: number | null;
      reps: number | null;
      distance: number | null;
      seconds: number | null;
      estimated_1rm: number | null;
    }[];
  }[];
}

interface VolumeSparklineData {
  exercise: string;
  date: string;
  volume: number;
  sets: number;
}

interface WorkoutDetailModalProps {
  isOpen: boolean;
  onClose: (open: boolean) => void;
  workout: WorkoutDetailData | null;
  isLoading: boolean;
  sparklineData?: VolumeSparklineData[];
}

function formatDuration(minutes: number | null): string {
  if (!minutes) return "";
  const hours = Math.floor(minutes / 60);
  const mins = Math.floor(minutes % 60);
  if (hours > 0) {
    return `${hours}h ${mins}m`;
  }
  return `${mins}m`;
}

function formatDateForModal(dateString: string): string {
  const date = new Date(dateString);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const isToday = date.toDateString() === today.toDateString();
  const isYesterday = date.toDateString() === yesterday.toDateString();

  if (isToday) return "Today";
  if (isYesterday) return "Yesterday";

  return date.toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

export default function WorkoutDetailModal({
  isOpen,
  onClose,
  workout,
  isLoading,
  sparklineData = [],
}: WorkoutDetailModalProps) {
  return (
    <Dialog
      open={isOpen}
      onOpenChange={onClose}
    >
      <DialogContent className='w-[95vw] sm:w-[90vw] md:w-[85vw] lg:max-w-6xl max-h-[80vh] overflow-hidden bg-background flex flex-col'>
        <DialogHeader className='sticky top-0 z-10 bg-background w-[95%] border-b border-border pb-4'>
          <DialogTitle className='text-xl font-bold w-fit'>
            {workout?.workout_name}
          </DialogTitle>
          <div className='text-sm text-muted-foreground flex flex-col items-start sm:flex-row sm:items-center gap-1 sm:gap-4'>
            <span>{formatDateForModal(workout?.date || "")}</span>
            {workout?.duration_minutes && (
              <span>Duration: {formatDuration(workout.duration_minutes)}</span>
            )}
          </div>
        </DialogHeader>

        <div className='flex-1 overflow-y-auto'>
          {isLoading ? (
            <div className='flex items-center justify-center py-8'>
              <div className='text-muted-foreground'>
                Loading workout details...
              </div>
            </div>
          ) : workout ? (
            <>
              <div
                className={cn(
                  "grid gap-4",
                  workout.exercises.length === 1
                    ? "grid-cols-1"
                    : workout.exercises.length === 2
                    ? "grid-cols-1 sm:grid-cols-2"
                    : "grid-cols-1 sm:grid-cols-2"
                )}
              >
                {workout.exercises
                  .map((exercise) => {
                    // Filter out placeholder/zero sets where all metrics are zero or null
                    const filteredSets = exercise.sets.filter((set) => {
                      const hasWeightReps =
                        set.weight !== null &&
                        set.reps !== null &&
                        (set.weight > 0 || set.reps > 0);
                      const hasDistance =
                        set.distance !== null && set.distance > 0;
                      const hasSeconds =
                        set.seconds !== null && set.seconds > 0;
                      const has1RM =
                        set.estimated_1rm !== null && set.estimated_1rm > 0;
                      return (
                        hasWeightReps || hasDistance || hasSeconds || has1RM
                      );
                    });
                    return { ...exercise, sets: filteredSets };
                  })
                  .filter((ex) => ex.sets.length > 0)
                  .map((exercise, exerciseIndex) => (
                    <div
                      key={exerciseIndex}
                      className='bg-card border rounded-lg p-3 sm:p-4'
                    >
                      <h3 className='font-semibold text-base sm:text-lg mb-3 sm:mb-4 break-words'>
                        {exercise.exercise}
                      </h3>
                      <div className='space-y-2'>
                        {exercise.sets.map((set, setIndex) => (
                          <div
                            key={setIndex}
                            className='flex items-center justify-between py-2 px-3 bg-muted/30 rounded-md'
                          >
                            <div className='flex items-center space-x-2 sm:space-x-4 min-w-0 flex-1'>
                              <span className='text-sm font-medium w-4 flex-shrink-0'>
                                {set.set_order}
                              </span>
                              <div className='flex items-center space-x-1 sm:space-x-2 min-w-0'>
                                {set.weight !== null && set.reps !== null && (
                                  <span className='text-sm truncate'>
                                    {set.weight} kg × {set.reps}
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className='flex items-center space-x-3 flex-shrink-0'>
                              {set.estimated_1rm && (
                                <div className='text-right'>
                                  <div className='text-xs text-muted-foreground'>
                                    1RM
                                  </div>
                                  <div className='text-sm font-medium'>
                                    {Math.round(set.estimated_1rm)}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>

              {/* Volume Sparklines Section */}
              {sparklineData.length > 0 && workout && (
                <div className='mt-8'>
                  <h3 className='text-lg font-semibold mb-4'>
                    Volume Trends, {workout.exercises.length} Workouts
                  </h3>
                  <div
                    className={cn(
                      "grid gap-4",
                      workout.exercises.length === 1
                        ? "grid-cols-1"
                        : workout.exercises.length === 2
                        ? "grid-cols-1 sm:grid-cols-2"
                        : workout.exercises.length === 3
                        ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
                        : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
                    )}
                  >
                    {workout.exercises.map((exercise, index) => {
                      // Filter to last 2 weeks (14 days)
                      const twoWeeksAgo = new Date();
                      twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);

                      const exerciseData = sparklineData
                        .filter(
                          (d) =>
                            d.exercise === exercise.exercise &&
                            new Date(d.date) >= twoWeeksAgo
                        )
                        .sort(
                          (a, b) =>
                            new Date(a.date).getTime() -
                            new Date(b.date).getTime()
                        );

                      if (exerciseData.length === 0) return null;

                      const latestValue =
                        exerciseData[exerciseData.length - 1]?.volume || 0;
                      const firstValue = exerciseData[0]?.volume || 0;
                      const delta =
                        firstValue > 0
                          ? ((latestValue - firstValue) / firstValue) * 100
                          : 0;

                      return (
                        <SparklineCard
                          key={index}
                          exercise={exercise.exercise}
                          data={exerciseData}
                          latestValue={latestValue}
                          delta={delta}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
