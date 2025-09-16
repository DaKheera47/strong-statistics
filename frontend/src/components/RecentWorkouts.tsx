"use client";

import { useEffect, useState } from "react";
import { WidgetHeader } from "./WidgetHeader";
import { WidgetWrapper } from "./WidgetWrapper";
import { AccordionContent } from "./ui/accordion";
import WorkoutDetailModal, { WorkoutDetailData } from "./WorkoutDetailModal";

interface RecentWorkoutData {
  date: string;
  workout_name: string;
  total_sets: number;
  total_volume: number;
  duration_minutes: number | null;
  exercises_count: number;
  prs: number;
}

interface VolumeSparklineData {
  exercise: string;
  date: string;
  volume: number;
  sets: number;
}


// WorkoutDetailData is now imported from WorkoutDetailModal

interface WorkoutCardProps {
  workout: RecentWorkoutData;
  workoutDetail: WorkoutDetailData | null;
  onTitleClick: () => void;
}

function formatDate(dateStr: string): { dayName: string; date: string } {
  const date = new Date(dateStr);
  const dayName = date.toLocaleDateString("en-US", { weekday: "long" });
  const formattedDate = date.toLocaleDateString("en-US", {
    day: "numeric",
    month: "short",
  });
  return { dayName, date: formattedDate };
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


function WorkoutCard({
  workout,
  workoutDetail,
  onTitleClick,
}: WorkoutCardProps) {
  const { dayName, date } = formatDate(workout.date);
  const duration = formatDuration(workout.duration_minutes);

  // Get exercises with their set counts and best sets
  const exercisesWithSets =
    workoutDetail?.exercises.map((exercise) => {
      const setCount = exercise.sets.length;
      const bestSet = exercise.sets.reduce((best, current) => {
        const bestScore = (best.weight || 0) * (best.reps || 0);
        const currentScore = (current.weight || 0) * (current.reps || 0);
        return currentScore > bestScore ? current : best;
      }, exercise.sets[0]);

      return {
        name: exercise.exercise,
        setCount,
        bestWeight: bestSet?.weight,
        bestReps: bestSet?.reps,
      };
    }) || [];

  return (
    <div className='bg-card rounded-lg p-6 border border-border'>
      <div className='flex justify-between items-start mb-4'>
        <div>
          <h3
            className='text-xl font-semibold text-foreground mb-1 cursor-pointer hover:text-primary transition-colors'
            onClick={onTitleClick}
          >
            {workout.workout_name}
          </h3>
          <p className='text-muted-foreground text-sm'>
            {dayName}, {date}
          </p>
        </div>
        <div className='text-blue-400'>
          <svg
            width='24'
            height='24'
            viewBox='0 0 24 24'
            fill='none'
            className='inline'
          >
            <circle
              cx='12'
              cy='12'
              r='3'
              fill='currentColor'
            />
            <circle
              cx='12'
              cy='12'
              r='7'
              stroke='currentColor'
              strokeWidth='2'
              fill='none'
            />
            <circle
              cx='12'
              cy='12'
              r='11'
              stroke='currentColor'
              strokeWidth='1'
              fill='none'
              opacity='0.5'
            />
          </svg>
        </div>
      </div>

      <div className='flex items-center gap-6 mb-6 text-sm'>
        <div className='flex items-center gap-1'>
          <svg
            width='16'
            height='16'
            viewBox='0 0 24 24'
            fill='none'
            className='text-muted-foreground'
          >
            <circle
              cx='12'
              cy='12'
              r='10'
              stroke='currentColor'
              strokeWidth='2'
            />
            <polyline
              points='12,6 12,12 16,14'
              stroke='currentColor'
              strokeWidth='2'
            />
          </svg>
          <span className='text-foreground'>{duration}</span>
        </div>
        <div className='flex items-center gap-1'>
          <svg
            width='16'
            height='16'
            viewBox='0 0 24 24'
            fill='none'
            className='text-muted-foreground'
          >
            <path
              d='M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z'
              stroke='currentColor'
              strokeWidth='2'
            />
          </svg>
          <span className='text-foreground'>
            {Math.round(workout.total_volume)} kg
          </span>
        </div>
        <div className='flex items-center gap-1'>
          <svg
            width='16'
            height='16'
            viewBox='0 0 24 24'
            fill='none'
            className='text-muted-foreground'
          >
            <polygon
              points='12,2 15.09,8.26 22,9 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9 8.91,8.26'
              stroke='currentColor'
              strokeWidth='2'
            />
          </svg>
          <span className='text-foreground'>{workout.prs} PRs</span>
        </div>
      </div>

      <div className='space-y-2'>
        <div className='flex justify-between text-sm font-medium text-muted-foreground mb-3'>
          <span>Exercise</span>
          <span>Best Set</span>
        </div>
        {exercisesWithSets.map((exercise, index) => {
          return (
            <div
              key={index}
              className='flex justify-between items-center'
            >
              <span className='text-foreground'>
                {exercise.setCount} × {exercise.name}
              </span>
              <span className='text-foreground'>
                {exercise.bestWeight
                  ? `${exercise.bestWeight} kg × ${exercise.bestReps}`
                  : "N/A"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function RecentWorkouts() {
  const [workouts, setWorkouts] = useState<RecentWorkoutData[]>([]);
  const [workoutDetails, setWorkoutDetails] = useState<
    Map<string, WorkoutDetailData>
  >(new Map());
  const [sparklineData, setSparklineData] = useState<VolumeSparklineData[]>([]);
  const [selectedWorkout, setSelectedWorkout] =
    useState<WorkoutDetailData | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchWorkouts() {
      try {
        // Fetch recent workouts list
        const response = await fetch("/api/recent-workouts");
        if (!response.ok) {
          throw new Error("Failed to fetch workouts");
        }

        const data: RecentWorkoutData[] = await response.json();
        const recentFour = data.slice(0, 4);
        setWorkouts(recentFour);

        // Fetch details for each workout
        const detailsMap = new Map<string, WorkoutDetailData>();

        for (const workout of recentFour) {
          try {
            const detailResponse = await fetch(
              `/api/recent-workouts?date=${encodeURIComponent(
                workout.date
              )}&workout_name=${encodeURIComponent(workout.workout_name)}`
            );
            if (detailResponse.ok) {
              const detail: WorkoutDetailData = await detailResponse.json();
              detailsMap.set(`${workout.date}-${workout.workout_name}`, detail);
            }
          } catch (err) {
            console.error(
              `Failed to fetch details for workout ${workout.workout_name}:`,
              err
            );
          }
        }

        setWorkoutDetails(detailsMap);

        // Fetch sparkline data
        try {
          const sparklineResponse = await fetch("/api/volume-sparklines");
          if (sparklineResponse.ok) {
            const sparklineData: VolumeSparklineData[] = await sparklineResponse.json();
            setSparklineData(sparklineData);
          }
        } catch (err) {
          console.error("Failed to fetch sparkline data:", err);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);
      }
    }

    fetchWorkouts();
  }, []);

  const fetchWorkoutDetail = async (date: string, workoutName: string) => {
    setDetailLoading(true);
    try {
      const response = await fetch(
        `/api/recent-workouts?date=${encodeURIComponent(
          date
        )}&workout_name=${encodeURIComponent(workoutName)}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch workout details");
      }
      const data: WorkoutDetailData = await response.json();
      setSelectedWorkout(data);
      setIsModalOpen(true);
    } catch (error) {
      console.error("Error fetching workout detail:", error);
    } finally {
      setDetailLoading(false);
    }
  };

  // formatDateForModal is now handled in WorkoutDetailModal

  if (loading) {
    return (
      <WidgetWrapper>
        <WidgetHeader
          title='Recent Workouts'
          isAccordion={true}
        />

        <AccordionContent className='space-y-4'>
          <div className='grid grid-cols-1 lg:grid-cols-2 gap-4'>
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className='bg-card rounded-lg p-6 border border-border animate-pulse'
              >
                <div className='h-6 bg-muted rounded w-1/3 mb-2'></div>
                <div className='h-4 bg-muted rounded w-1/4 mb-4'></div>
                <div className='flex gap-6 mb-6'>
                  <div className='h-4 bg-muted rounded w-12'></div>
                  <div className='h-4 bg-muted rounded w-16'></div>
                  <div className='h-4 bg-muted rounded w-12'></div>
                </div>
                <div className='space-y-2'>
                  {[...Array(3)].map((_, j) => (
                    <div
                      key={j}
                      className='flex justify-between'
                    >
                      <div className='h-4 bg-muted rounded w-1/2'></div>
                      <div className='h-4 bg-muted rounded w-1/4'></div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </AccordionContent>
      </WidgetWrapper>
    );
  }

  if (error) {
    return (
      <WidgetWrapper>
        <WidgetHeader
          title='Recent Workouts'
          isAccordion={true}
        />
        <AccordionContent className='space-y-4'>
          <div className='bg-card rounded-lg p-6 border border-border text-center'>
            <p className='text-muted-foreground'>
              Failed to load recent workouts: {error}
            </p>
          </div>
        </AccordionContent>
      </WidgetWrapper>
    );
  }

  if (workouts.length === 0) {
    return (
      <WidgetWrapper>
        <WidgetHeader
          title='Recent Workouts'
          isAccordion={true}
        />
        <AccordionContent className='space-y-4'>
          <div className='bg-card rounded-lg p-6 border border-border text-center'>
            <p className='text-muted-foreground'>No recent workouts found</p>
          </div>
        </AccordionContent>
      </WidgetWrapper>
    );
  }

  return (
    <>
      <WidgetWrapper>
        <WidgetHeader
          title='Recent Workouts'
          isAccordion={true}
        />
        <AccordionContent className='space-y-4'>
          <div className='grid grid-cols-1 gap-4'>
            {workouts.map((workout) => {
              const key = `${workout.date}-${workout.workout_name}`;
              const detail = workoutDetails.get(key);

              return (
                <WorkoutCard
                  key={key}
                  workout={workout}
                  workoutDetail={detail || null}
                  onTitleClick={() =>
                    fetchWorkoutDetail(workout.date, workout.workout_name)
                  }
                />
              );
            })}
          </div>
        </AccordionContent>
      </WidgetWrapper>

      <WorkoutDetailModal
        isOpen={isModalOpen}
        onClose={setIsModalOpen}
        workout={selectedWorkout}
        isLoading={detailLoading}
        sparklineData={sparklineData}
      />
    </>
  );
}
