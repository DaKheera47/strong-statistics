'use client';

import { useMemo, useEffect, useState } from 'react';
import { useRepRangeDistribution } from '@/hooks/useRepRangeDistribution';
import { BarChart, Bar, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useChartColorsArray } from '@/hooks/useChartColors';
import { ExerciseFilter, getRecentExercises } from './exercise-filter';
import { useExerciseSelection } from '@/hooks/useExerciseSelection';
import { WidgetWrapper } from './WidgetWrapper';
import { WidgetHeader } from './WidgetHeader';
import { AccordionContent } from './ui/accordion';
import { Button } from './ui/button';
import { shouldDisplayDistance, getDistanceUnit } from '@/lib/exercise-config';
import { getRepRangeDistributionLimit } from '@/lib/localStorage';

interface RepRangeCardProps {
  exercise: string;
  data: Array<{
    date: string;
    range_1_5: number;
    range_6_12: number;
    range_13_20: number;
    range_20_plus: number;
    total: number;
  }>;
}

function RepRangeCard({ exercise, data }: RepRangeCardProps) {
  const chartColors = useChartColorsArray();
  const showDistance = shouldDisplayDistance(exercise);
  const distanceUnit = getDistanceUnit(exercise);
  
  // Use chart colors for each rep range (or distance range for kettlebell carries)
  const repRangeColors = {
    range_1_5: chartColors[4] || '#ef4444',      // Chart 5 - Heavy/Strength
    range_6_12: chartColors[0] || '#3b82f6',     // Chart 1 - Hypertrophy 
    range_13_20: chartColors[1] || '#10b981',    // Chart 2 - Endurance
    range_20_plus: chartColors[3] || '#f59e0b'   // Chart 4 - High endurance
  };

  // Filter out days with no sets
  const filteredData = data.filter(d => d.total > 0);

  if (filteredData.length === 0) {
    return (
      <div className="p-4 border rounded-lg bg-card">
        <h3 className="font-medium text-sm text-card-foreground mb-4">
          {exercise}
        </h3>
        <div className="text-muted-foreground text-sm text-center py-8">
          No data available
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 border rounded-lg bg-card hover:shadow-md transition-shadow">
      <h3 className="font-medium text-sm text-card-foreground mb-4 truncate">
        {exercise}
      </h3>
      
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={filteredData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 10 }}
              tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip 
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const total = payload.reduce((sum, p) => sum + (p.value as number), 0);
                  return (
                    <div className="bg-popover p-3 border rounded shadow-lg">
                      <p className="text-sm font-semibold text-popover-foreground mb-2">
                        {new Date(label ?? "").toLocaleDateString()}
                      </p>
                      {payload.map((p, index) => (
                        <div key={index} className="flex justify-between items-center text-xs mb-1">
                          <span style={{ color: p.color }}>
                            {showDistance ? (
                              p.dataKey === 'range_1_5' ? `1-5 ${distanceUnit}` :
                              p.dataKey === 'range_6_12' ? `6-12 ${distanceUnit}` :
                              p.dataKey === 'range_13_20' ? `13-20 ${distanceUnit}` :
                              `20+ ${distanceUnit}`
                            ) : (
                              p.dataKey === 'range_1_5' ? '1-5 reps' :
                              p.dataKey === 'range_6_12' ? '6-12 reps' :
                              p.dataKey === 'range_13_20' ? '13-20 reps' :
                              '20+ reps'
                            )}:
                          </span>
                          <span className="ml-2 font-medium">{p.value} sets</span>
                        </div>
                      ))}
                      <div className="border-t pt-1 mt-2 text-xs font-semibold">
                        Total: {total} sets
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="range_1_5" stackId="a" fill={repRangeColors.range_1_5} />
            <Bar dataKey="range_6_12" stackId="a" fill={repRangeColors.range_6_12} />
            <Bar dataKey="range_13_20" stackId="a" fill={repRangeColors.range_13_20} />
            <Bar dataKey="range_20_plus" stackId="a" fill={repRangeColors.range_20_plus} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-3 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: repRangeColors.range_1_5 }}></div>
          <span className="text-muted-foreground">{showDistance ? `1-5 ${distanceUnit}` : '1-5 reps'}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: repRangeColors.range_6_12 }}></div>
          <span className="text-muted-foreground">{showDistance ? `6-12 ${distanceUnit}` : '6-12 reps'}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: repRangeColors.range_13_20 }}></div>
          <span className="text-muted-foreground">{showDistance ? `13-20 ${distanceUnit}` : '13-20 reps'}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: repRangeColors.range_20_plus }}></div>
          <span className="text-muted-foreground">{showDistance ? `20+ ${distanceUnit}` : '20+ reps'}</span>
        </div>
      </div>
    </div>
  );
}

export default function RepRangeDistribution() {
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

  const limit = getRepRangeDistributionLimit();
  
  // === Rep-based exercise filtering ===
  // We only show / allow exercises that are rep-based for this widget since the distribution is reps-based.
  const repBasedExercises = useMemo(() => {
    return allExercises.filter(ex => !shouldDisplayDistance(ex.name));
  }, [allExercises]);

  // Sanitize currently selected exercises so distance-based ones are silently removed.
  const sanitizedGlobalSelection = useMemo(() => {
    if (globalSelectedExercises.length === 0) return globalSelectedExercises;
    const repNames = new Set(repBasedExercises.map(e => e.name));
    return globalSelectedExercises.filter(name => repNames.has(name));
  }, [globalSelectedExercises, repBasedExercises]);

  // Initialize local selection with limited exercises
  useEffect(() => {
    if (!initialized && repBasedExercises.length > 0 && sanitizedGlobalSelection.length > 0) {
      // If more than 10 exercises are selected globally, limit to default
      const limitedSelection = sanitizedGlobalSelection.length > 10
        ? sanitizedGlobalSelection.slice(0, limit)
        : sanitizedGlobalSelection;
      setLocalSelectedExercises(limitedSelection);
      setInitialized(true);
    }
  }, [repBasedExercises, sanitizedGlobalSelection, limit, initialized]);

  // Effect to commit sanitized selection if it differs (avoids re-render loop by shallow compare)
  useEffect(() => {
    if (sanitizedGlobalSelection.length !== globalSelectedExercises.length) {
      // Distance-based exercises were removed
      setGlobalSelectedExercises(sanitizedGlobalSelection);
    } else {
      // Possible case: same length but different ordering after filtering (unlikely); ensure stable ordering of original selection
      const changed = sanitizedGlobalSelection.some((n, i) => n !== globalSelectedExercises[i]);
      if (changed) {
        setGlobalSelectedExercises(sanitizedGlobalSelection);
      }
    }
  }, [sanitizedGlobalSelection, globalSelectedExercises, setGlobalSelectedExercises]);

  // Get filtered exercise list based on recent filter (memoized)
  const filteredRepBasedExercises = useMemo(() => {
    if (!showRecentOnly) return repBasedExercises;
    const recentExerciseNames = getRecentExercises(allExercises);
    return repBasedExercises.filter(ex => recentExerciseNames.includes(ex.name));
  }, [showRecentOnly, repBasedExercises, allExercises]);

  // Memoize selected exercises to prevent render loops
  const selectedExercises = useMemo(() => {
    return showRecentOnly
      ? filteredRepBasedExercises.map(ex => ex.name)
      : localSelectedExercises;
  }, [showRecentOnly, filteredRepBasedExercises, localSelectedExercises]);


  const handleToggleRecentOnly = () => {
    setShowRecentOnly(!showRecentOnly);
    if (!showRecentOnly) {
      // When turning on recent filter, it will auto-select all recent exercises via selectedExercises logic
      // No need to manually update selection here
    } else {
      // When turning off recent filter, reset to limited selection
      const currentSelection = filteredRepBasedExercises.map(ex => ex.name);
      if (currentSelection.length > limit) {
        const limitedSelection = currentSelection.slice(0, limit);
        setLocalSelectedExercises(limitedSelection);
      } else {
        setLocalSelectedExercises(currentSelection);
      }
    }
  };

  // Handler to ensure any attempted additions of distance-based exercises are ignored
  const handleSelectionChange = useMemo(() => (newSelection: string[]) => {
    const repNames = new Set(repBasedExercises.map(e => e.name));
    const filtered = newSelection.filter(n => repNames.has(n));

    if (showRecentOnly) {
      // When in recent mode, update global selection but don't limit
      setGlobalSelectedExercises(filtered);
    } else {
      // When in normal mode, update both local and global
      setLocalSelectedExercises(filtered);
      setGlobalSelectedExercises(filtered);
    }
  }, [repBasedExercises, setGlobalSelectedExercises, showRecentOnly]);
  
  const { data, loading, error } = useRepRangeDistribution(selectedExercises);

  if (loading) {
    return (
      <WidgetWrapper>
        <WidgetHeader title="Rep Range Distribution">
          <ExerciseFilter
            allExercises={filteredRepBasedExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder="Filter exercises..."
          />
        </WidgetHeader>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="p-4 border rounded-lg bg-muted animate-pulse">
              <div className="h-4 bg-muted-foreground/20 rounded mb-4"></div>
              <div className="h-48 bg-muted-foreground/20 rounded"></div>
            </div>
          ))}
        </div>
      </WidgetWrapper>
    );
  }

  if (error) {
    return (
      <WidgetWrapper>
        <WidgetHeader title="Rep Range Distribution">
          <ExerciseFilter
            allExercises={filteredRepBasedExercises}
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

  const filteredData = data.filter((exerciseData) => !shouldDisplayDistance(exerciseData.exercise));
  
  if (filteredData.length === 0) {
    return (
      <WidgetWrapper>
        <WidgetHeader title="Rep Range Distribution">
          <ExerciseFilter
            allExercises={filteredRepBasedExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={handleSelectionChange}
            loading={exercisesLoading}
            error={exercisesError}
            placeholder="Filter exercises..."
          />
        </WidgetHeader>
        <div className="text-muted-foreground bg-muted p-4 rounded-lg text-center">
          No rep range data available for selected exercises.
        </div>
      </WidgetWrapper>
    );
  }

  return (
    <WidgetWrapper>
      <WidgetHeader title="Rep Range Distribution" isAccordion>
        <div className="flex flex-col sm:flex-row gap-2 sm:items-center">
          {/* For this widget we intentionally restrict to rep-based exercises only */}
          <ExerciseFilter
            allExercises={filteredRepBasedExercises}
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
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {filteredData.map((exerciseData) => (
            <RepRangeCard
              key={exerciseData.exercise}
              exercise={exerciseData.exercise}
              data={exerciseData.data}
            />
          ))}
        </div>
      </AccordionContent>
    </WidgetWrapper>
  );
}