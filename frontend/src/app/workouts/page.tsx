"use client";

import { useEffect, useState } from "react";

import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { RecentWorkoutData } from "../api/recent-workouts/route";
import { WorkoutDetailData } from "@/components/WorkoutDetailModal";
import WorkoutDetailModalWithSparklines from "@/components/WorkoutDetailModalWithSparklines";

export default function WorkoutsPage() {
  const [workouts, setWorkouts] = useState<RecentWorkoutData[]>([]);
  const [selectedWorkout, setSelectedWorkout] =
    useState<WorkoutDetailData | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchWorkouts();
  }, []);

  const fetchWorkouts = async () => {
    try {
      const response = await fetch("/api/recent-workouts");
      const data = await response.json();
      setWorkouts(data);
    } catch (error) {
      console.error("Error fetching workouts:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchWorkoutDetail = async (date: string, workoutName: string) => {
    setDetailLoading(true);
    try {
      const response = await fetch(
        `/api/recent-workouts?date=${date}&workout_name=${encodeURIComponent(
          workoutName
        )}`
      );
      const data = await response.json();
      setSelectedWorkout(data);
      setIsModalOpen(true);
    } catch (error) {
      console.error("Error fetching workout detail:", error);
    } finally {
      setDetailLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
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
  };

  const formatDuration = (minutes: number | null) => {
    if (minutes === null || minutes === undefined) return "-";
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    if (hours > 0) {
      return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
    }
    return `${mins}m`;
  };

  if (loading) {
    return (
      <div className='min-h-screen bg-background flex items-center justify-center'>
        <div className='text-muted-foreground'>Loading workouts...</div>
      </div>
    );
  }

  return (
    <div className='min-h-screen bg-background'>
      <div className='py-8'>
        <header className='mb-8'>
          <div className='flex justify-between items-start mb-4'>
            <div>
              <h1 className='text-3xl font-bold text-foreground mb-2'>
                Recent Workouts
              </h1>
              <p className='text-muted-foreground'>
                View and analyze your training history
              </p>
            </div>
            <div className='flex items-center gap-4'>
              <Link href='/'>
                <Button variant='outline'>Back to Dashboard</Button>
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main>
          <div className='bg-card border rounded-lg overflow-hidden'>
            <div className='overflow-x-auto'>
              <table className='w-full'>
                <thead className='bg-muted/50'>
                  <tr className='border-b'>
                    <th className='text-left p-4 font-semibold text-sm'>
                      Date
                    </th>
                    <th className='text-left p-4 font-semibold text-sm'>
                      Workout
                    </th>
                    <th className='text-left p-4 font-semibold text-sm'>
                      Duration
                    </th>
                    <th className='text-left p-4 font-semibold text-sm'>
                      Sets
                    </th>
                    <th className='text-left p-4 font-semibold text-sm'>
                      Volume (kg)
                    </th>
                    <th className='text-left p-4 font-semibold text-sm'>
                      Exercises
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {workouts.map((workout) => (
                    <tr
                      key={`${workout.date}-${workout.workout_name}`}
                      className='border-b hover:bg-muted/30 cursor-pointer transition-colors'
                      onClick={() =>
                        fetchWorkoutDetail(workout.date, workout.workout_name)
                      }
                    >
                      <td className='p-4'>
                        <div className='flex flex-col'>
                          <span className='font-medium text-sm'>
                            {formatDate(workout.date)}
                          </span>
                          <span className='text-xs text-muted-foreground'>
                            {new Date(workout.date).toLocaleDateString()}
                          </span>
                        </div>
                      </td>
                      <td className='p-4'>
                        <span className='font-medium text-sm'>
                          {workout.workout_name}
                        </span>
                      </td>
                      <td className='p-4'>
                        <span className='text-sm'>
                          {formatDuration(workout.duration_minutes)}
                        </span>
                      </td>
                      <td className='p-4'>
                        <span className='text-sm font-mono'>
                          {workout.total_sets}
                        </span>
                      </td>
                      <td className='p-4'>
                        <span className='text-sm font-mono'>
                          {Math.round(workout.total_volume).toLocaleString()}
                        </span>
                      </td>
                      <td className='p-4'>
                        <span className='text-sm'>
                          {workout.exercises_count}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>

      <WorkoutDetailModalWithSparklines
        isOpen={isModalOpen}
        onClose={setIsModalOpen}
        workout={selectedWorkout}
        isLoading={detailLoading}
      />
    </div>
  );
}
