"use client";

import { useVolumeData } from "@/hooks/useVolumeData";
import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { cn } from "@/lib/utils";
import { useChartColors } from "@/hooks/useChartColors";
import { ExerciseFilter } from "./exercise-filter";
import { useExerciseSelection } from "@/hooks/useExerciseSelection";
import { WidgetWrapper } from "./WidgetWrapper";
import { WidgetHeader } from "./WidgetHeader";
import { AccordionContent } from "./ui/accordion";

interface SparklineCardProps {
  exercise: string;
  data: Array<{
    date: string;
    volume: number;
    sets: number;
  }>;
  latestValue: number;
  delta: number;
}

function SparklineCard({
  exercise,
  data,
  latestValue,
  delta,
}: SparklineCardProps) {
  const colors = useChartColors();
  // Delta is already calculated correctly based on exercise type in the hook
  const deltaColor =
    delta > 0
      ? "text-green-600 dark:text-green-400"
      : delta < 0
      ? "text-red-600 dark:text-red-400"
      : "text-muted-foreground";
  const deltaSign = delta > 0 ? "+" : "";

  return (
    <div className='p-4 border rounded-lg bg-card hover:shadow-md transition-shadow'>
      <div className='flex justify-between items-start mb-2'>
        <h3 className='font-medium text-sm text-card-foreground truncate flex-1 mr-2'>
          {exercise}
        </h3>
        <div className='text-right'>
          <div className='text-lg font-semibold text-card-foreground'>
            {Math.round(latestValue)}
          </div>
          {delta !== 0 && (
            <div className={cn("text-xs", deltaColor)}>
              {deltaSign}
              {delta.toFixed(1)}%
            </div>
          )}
        </div>
      </div>

      <div className='h-20 w-full'>
        <ResponsiveContainer
          width='100%'
          height='100%'
        >
          <LineChart
            data={data}
            margin={{ left: -5, right: 5, top: 5, bottom: 5 }}
          >
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "currentColor" }}
              domain={["dataMin - 50", "dataMax + 50"]}
              width={35}
            />
            <Line
              type='monotone'
              dataKey='volume'
              stroke={colors.primary}
              strokeWidth={1.5}
              dot={{ fill: colors.primary, r: 2 }}
              activeDot={{
                r: 3,
                stroke: colors.primary,
                strokeWidth: 2,
                fill: colors.background,
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className='bg-popover p-2 border rounded shadow-lg'>
                      <p className='text-xs text-muted-foreground'>
                        {new Date(data.date).toLocaleDateString()}
                      </p>
                      <p className='text-sm font-semibold text-popover-foreground'>
                        Volume: {Math.round(data.volume)} kg
                      </p>
                      <p className='text-xs text-muted-foreground'>
                        Sets: {data.sets}
                      </p>
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

export default function VolumeSparklines() {
  const {
    allExercises,
    selectedExercises,
    setSelectedExercises,
    loading: exercisesLoading,
    error: exercisesError,
  } = useExerciseSelection();

  const { data, loading, error } = useVolumeData(selectedExercises);

  if (loading) {
    return (
      <WidgetWrapper>
        <WidgetHeader title="Volume Sparklines">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={setSelectedExercises}
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
        <WidgetHeader title="Volume Sparklines">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={setSelectedExercises}
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
        <WidgetHeader title="Volume Sparklines">
          <ExerciseFilter
            allExercises={allExercises}
            selectedExercises={selectedExercises}
            onSelectionChange={setSelectedExercises}
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
      <WidgetHeader title="Volume Sparklines" isAccordion>
        <ExerciseFilter
          allExercises={allExercises}
          selectedExercises={selectedExercises}
          onSelectionChange={setSelectedExercises}
          loading={exercisesLoading}
          error={exercisesError}
          placeholder='Filter exercises...'
        />
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
