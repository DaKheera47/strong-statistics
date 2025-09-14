"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";

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

interface WorkoutDetailModalProps {
  isOpen: boolean;
  onClose: (open: boolean) => void;
  workout: WorkoutDetailData | null;
  isLoading: boolean;
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
}: WorkoutDetailModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="w-[95vw] sm:w-[90vw] md:w-[85vw] lg:max-w-4xl max-h-[80vh] overflow-y-auto bg-background">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold">
            {workout?.workout_name}
          </DialogTitle>
          <div className="text-sm text-muted-foreground flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
            <span>{formatDateForModal(workout?.date || "")}</span>
            {workout?.duration_minutes && (
              <span>
                Duration: {formatDuration(workout.duration_minutes)}
              </span>
            )}
          </div>
        </DialogHeader>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-muted-foreground">
              Loading workout details...
            </div>
          </div>
        ) : workout ? (
          <div className="space-y-6 mt-4">
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
                  const hasSeconds = set.seconds !== null && set.seconds > 0;
                  const has1RM =
                    set.estimated_1rm !== null && set.estimated_1rm > 0;
                  return hasWeightReps || hasDistance || hasSeconds || has1RM;
                });
                return { ...exercise, sets: filteredSets };
              })
              .filter((ex) => ex.sets.length > 0)
              .map((exercise, exerciseIndex) => (
                <div
                  key={exerciseIndex}
                  className="bg-card border rounded-lg p-3 sm:p-4"
                >
                  <h3 className="font-semibold text-base sm:text-lg mb-3 sm:mb-4 break-words">
                    {exercise.exercise}
                  </h3>
                  <div className="space-y-2">
                    {exercise.sets.map((set, setIndex) => (
                      <div
                        key={setIndex}
                        className="flex items-center justify-between py-2 px-3 bg-muted/30 rounded-md"
                      >
                        <div className="flex items-center space-x-2 sm:space-x-4 min-w-0 flex-1">
                          <span className="text-sm font-medium w-4 flex-shrink-0">
                            {set.set_order}
                          </span>
                          <div className="flex items-center space-x-1 sm:space-x-2 min-w-0">
                            {set.weight !== null && set.reps !== null && (
                              <span className="text-sm truncate">
                                {set.weight} kg × {set.reps}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center space-x-3 flex-shrink-0">
                          {set.estimated_1rm && (
                            <div className="text-right">
                              <div className="text-xs text-muted-foreground">
                                1RM
                              </div>
                              <div className="text-sm font-medium">
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
        ) : null}
      </DialogContent>
    </Dialog>
  );
}