'use client';

import { useState, useEffect, useMemo } from 'react';
import { useMaxWeightData } from '@/hooks/useMaxWeightData';
import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { cn } from '@/lib/utils';
import { useChartColors } from '@/hooks/useChartColors';
import { ExerciseFilter, getRecentExercises } from './exercise-filter';
import { useExerciseSelection } from '@/hooks/useExerciseSelection';
import { WidgetWrapper } from './WidgetWrapper';
import { WidgetHeader } from './WidgetHeader';
import { AccordionContent } from './ui/accordion';
import { Button } from './ui/button';
import { isLowerBetter, shouldDisplayDistance, getDistanceUnit } from '@/lib/exercise-config';
import { getMaxWeightSparklineLimit } from '@/lib/localStorage';

interface SparklineCardProps {
  exercise: string;
  data: Array<{
    date: string;
    maxWeight: number;
    reps: number | null;
    distance: number | null;
  }>;
  latestValue: number;
  delta: number;
}

function SparklineCard({ exercise, data, latestValue, delta }: SparklineCardProps) {
  const colors = useChartColors();
  const lowerIsBetter = isLowerBetter(exercise);
  const showDistance = shouldDisplayDistance(exercise);
  const distanceUnit = getDistanceUnit(exercise);
  
  // Delta color logic (delta is already calculated correctly based on exercise type)
  const deltaColor = delta > 0 ? 'text-green-600 dark:text-green-400' : delta < 0 ? 'text-red-600 dark:text-red-400' : 'text-muted-foreground';
  
  const deltaSign = delta > 0 ? '+' : '';

  return (
    <div className="p-4 border rounded-lg bg-card hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-medium text-sm text-card-foreground truncate flex-1 mr-2">
          {exercise}
        </h3>
        <div className="text-right">
          <div className="text-lg font-semibold text-card-foreground">
            {Math.round(latestValue)} kg
          </div>
          {delta !== 0 && (
            <div className={cn("text-xs", deltaColor)}>
              {deltaSign}{delta.toFixed(1)}%
            </div>
          )}
        </div>
      </div>
      
      <div className="h-20 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: -5, right: 5, top: 5, bottom: 5 }}>
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: 'currentColor' }}
              domain={['dataMin - 5', 'dataMax + 5']}
              width={35}
            />
            <Line 
              type="monotone" 
              dataKey="maxWeight" 
              stroke={colors.secondary} 
              strokeWidth={1.5}
              dot={{ fill: colors.secondary, r: 2 }}
              activeDot={{ r: 3, stroke: colors.secondary, strokeWidth: 2, fill: colors.background }}
            />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const point = payload[0].payload;
                  // For distance-based exercises we currently want to surface the reps value AS the distance value (no separate reps line)
                  // because distance data may be unreliable/missing. Use reps as a proxy distance count.
                  const repsDisplay = point.reps != null ? point.reps : '—';
                  const distanceDisplay = showDistance ? `${repsDisplay} ${distanceUnit}` : null;
                  return (
                    <div className="bg-popover p-2 border rounded shadow-lg">
                      <p className="text-xs text-muted-foreground">
                        {new Date(point.date).toLocaleDateString()}
                      </p>
                      <p className="text-sm font-semibold text-popover-foreground">
                        Max Weight: {Math.round(point.maxWeight)} kg
                      </p>
                      {showDistance ? (
                        <p className="text-xs text-muted-foreground">Distance: {distanceDisplay}</p>
                      ) : (
                        <p className="text-xs text-muted-foreground">Reps: {repsDisplay}</p>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function MaxWeightSparklines() {
  const {
    allExercises,
    selectedExercises: globalSelectedExercises,
    setSelectedExercises: setGlobalSelectedExercises,
    loading: exercisesLoading,
    error: exercisesError
  } = useExerciseSelection();

  const [localSelectedExercises, setLocalSelectedExercises] = useState<string[]>([]);
  const [showRecentOnly, setShowRecentOnly] = useState(false);
  const [initialized, setInitialized] = useState(false);

  const limit = getMaxWeightSparklineLimit();

  // Initialize local selection with limited exercises
  useEffect(() => {
    if (!initialized && allExercises.length > 0 && globalSelectedExercises.length > 0) {
      // If more than 10 exercises are selected globally, limit to default
      const limitedSelection = globalSelectedExercises.length > 10
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
    return allExercises.filter(ex => recentExerciseNames.includes(ex.name));
  }, [showRecentOnly, allExercises]);

  // Memoize selected exercises to prevent render loops
  const selectedExercises = useMemo(() => {
    return showRecentOnly
      ? filteredAllExercises.map(ex => ex.name)
      : localSelectedExercises;
  }, [showRecentOnly, filteredAllExercises, localSelectedExercises]);

  const { data, loading, error } = useMaxWeightData(selectedExercises);


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
        <WidgetHeader title="Max Weight Sparklines">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder="Filter exercises..."
          />
        </WidgetHeader>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="p-4 border rounded-lg bg-muted animate-pulse">
              <div className="h-4 bg-muted-foreground/20 rounded mb-2"></div>
              <div className="h-16 bg-muted-foreground/20 rounded"></div>
            </div>
          ))}
        </div>
      </WidgetWrapper>
    );
  }

  if (error) {
    return (
      <WidgetWrapper>
        <WidgetHeader title="Max Weight Sparklines">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder="Filter exercises..."
          />
        </WidgetHeader>
        <div className="text-destructive bg-destructive/10 p-4 rounded-lg">
          Error loading data: {error}
        </div>
      </WidgetWrapper>
    );
  }

  if (data.length === 0) {
    return (
      <WidgetWrapper>
        <WidgetHeader title="Max Weight Sparklines">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder="Filter exercises..."
          />
        </WidgetHeader>
        <div className="text-muted-foreground bg-muted p-4 rounded-lg text-center">
          No training data available for selected exercises.
        </div>
      </WidgetWrapper>
    );
  }

  return (
    <WidgetWrapper>
      <WidgetHeader title="Max Weight Sparklines" isAccordion>
        <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder="Filter exercises..."
          />
          <Button
            variant={showRecentOnly ? "default" : "outline"}
            size="sm"
            onClick={handleToggleRecentOnly}
          >
            Recent Only
          </Button>
        </div>
      </WidgetHeader>

      <AccordionContent>
        <div className={cn(
          "grid gap-4",
          data.length === 1 ? "grid-cols-1" :
          data.length === 2 ? "grid-cols-1 sm:grid-cols-2" :
          data.length === 3 ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3" :
          "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        )}>
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