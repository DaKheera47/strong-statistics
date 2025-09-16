"use client";

import { useExerciseSelection } from "@/hooks/useExerciseSelection";
import { useVolumeData } from "@/hooks/useVolumeData";
import { getVolumeSparklineLimit } from "@/lib/localStorage";
import { cn } from "@/lib/utils";
import { useEffect, useMemo, useState } from "react";
import { SparklineCard } from "./SparklineCard";
import { WidgetHeader } from "./WidgetHeader";
import { WidgetWrapper } from "./WidgetWrapper";
import { ExerciseFilter, getRecentExercises } from "./exercise-filter";
import { AccordionContent } from "./ui/accordion";
import { Button } from "./ui/button";

export default function VolumeSparklines() {
  const {
    allExercises,
    selectedExercises: globalSelectedExercises,
    setSelectedExercises: setGlobalSelectedExercises,
    loading: exercisesLoading,
    error: exercisesError,
  } = useExerciseSelection();

  const [localSelectedExercises, setLocalSelectedExercises] = useState<
    string[]
  >([]);
  const [showRecentOnly, setShowRecentOnly] = useState(false);
  const [initialized, setInitialized] = useState(false);

  const limit = getVolumeSparklineLimit();

  // Initialize local selection with limited exercises
  useEffect(() => {
    if (
      !initialized &&
      allExercises.length > 0 &&
      globalSelectedExercises.length > 0
    ) {
      // If more than 10 exercises are selected globally, limit to default
      const limitedSelection =
        globalSelectedExercises.length > 10
          ? globalSelectedExercises.slice(0, limit)
          : globalSelectedExercises;
      setLocalSelectedExercises(limitedSelection);
      setInitialized(true);
    }
  }, [allExercises, globalSelectedExercises, limit, initialized]);

  // Get filtered exercise list based on recent filter (memoized)
  const filteredAllExercises = useMemo(() => {
    if (!showRecentOnly) return allExercises;
    const recentExerciseNames = getRecentExercises(allExercises);
    return allExercises.filter((ex) => recentExerciseNames.includes(ex.name));
  }, [showRecentOnly, allExercises]);

  // Memoize selected exercises to prevent render loops
  const selectedExercises = useMemo(() => {
    return showRecentOnly
      ? filteredAllExercises.map((ex) => ex.name)
      : localSelectedExercises;
  }, [showRecentOnly, filteredAllExercises, localSelectedExercises]);

  const { data, loading, error } = useVolumeData(selectedExercises);

  const handleToggleRecentOnly = () => {
    setShowRecentOnly(!showRecentOnly);
    if (!showRecentOnly) {
      // When turning on recent filter, it will auto-select all recent exercises via selectedExercises logic
      // No need to manually update selection here
    } else {
      // When turning off recent filter, reset to limited selection
      if (globalSelectedExercises.length > limit) {
        const limitedSelection = globalSelectedExercises.slice(0, limit);
        setLocalSelectedExercises(limitedSelection);
      } else {
        setLocalSelectedExercises(globalSelectedExercises);
      }
    }
  };

  const handleSelectionChange = (newSelection: string[]) => {
    if (showRecentOnly) {
      // When in recent mode, update global selection but don't limit
      setGlobalSelectedExercises(newSelection);
    } else {
      // When in normal mode, update both local and global
      setLocalSelectedExercises(newSelection);
      setGlobalSelectedExercises(newSelection);
    }
  };

  if (loading) {
    return (
      <WidgetWrapper>
        <WidgetHeader title='Volume Sparklines'>
          <ExerciseFilter
            allExercises={filteredAllExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder='Filter exercises...'
          />
        </WidgetHeader>
        <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4'>
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className='p-4 border rounded-lg bg-muted animate-pulse'
            >
              <div className='h-4 bg-muted-foreground/20 rounded mb-2'></div>
              <div className='h-16 bg-muted-foreground/20 rounded'></div>
            </div>
          ))}
        </div>
      </WidgetWrapper>
    );
  }

  if (error) {
    return (
      <WidgetWrapper>
        <WidgetHeader title='Volume Sparklines'>
          <ExerciseFilter
            allExercises={filteredAllExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder='Filter exercises...'
          />
        </WidgetHeader>
        <div className='text-destructive bg-destructive/10 p-4 rounded-lg'>
          Error loading data: {error}
        </div>
      </WidgetWrapper>
    );
  }

  if (data.length === 0) {
    return (
      <WidgetWrapper>
        <WidgetHeader title='Volume Sparklines'>
          <ExerciseFilter
            allExercises={filteredAllExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder='Filter exercises...'
          />
        </WidgetHeader>
        <div className='text-muted-foreground bg-muted p-4 rounded-lg text-center'>
          No training data available for selected exercises.
        </div>
      </WidgetWrapper>
    );
  }

  return (
    <WidgetWrapper>
      <WidgetHeader
        title='Volume Sparklines'
        isAccordion
      >
        <div className='flex flex-col sm:flex-row gap-2 sm:items-center'>
          <ExerciseFilter
            allExercises={filteredAllExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder='Filter exercises...'
          />
          <Button
            variant={showRecentOnly ? "default" : "outline"}
            size='sm'
            onClick={handleToggleRecentOnly}
          >
            Recent Only
          </Button>
        </div>
      </WidgetHeader>

      <AccordionContent>
        <div
          className={cn(
            "grid gap-4",
            data.length === 1
              ? "grid-cols-1"
              : data.length === 2
              ? "grid-cols-1 sm:grid-cols-2"
              : data.length === 3
              ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
              : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
          )}
        >
          {data.map((exerciseData) => (
            <SparklineCard
              key={exerciseData.exercise}
              exercise={exerciseData.exercise}
              data={exerciseData.data}
              latestValue={exerciseData.latestValue}
              delta={exerciseData.delta}
            />
          ))}
        </div>
      </AccordionContent>
    </WidgetWrapper>
  );
}
