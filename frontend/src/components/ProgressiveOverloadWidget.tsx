"use client";

import { useChartColors, useChartColorsArray } from "@/hooks/useChartColors";
import { useExerciseSelection } from "@/hooks/useExerciseSelection";
import {
  ProgressiveOverloadDataPoint,
  ProgressiveVolumeDataPoint,
  useProgressiveOverloadData,
} from "@/hooks/useProgressiveOverloadData";
import { shouldDisplayDistance, getDistanceUnit } from "@/lib/exercise-config";
import { useEffect, useMemo, useState } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { WidgetHeader } from "./WidgetHeader";
import { WidgetWrapper } from "./WidgetWrapper";
import { AccordionContent } from "./ui/accordion";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "./ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "./ui/popover";
import { Button } from "./ui/button";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { RadioGroup, RadioGroupItem } from "./ui/radio-group";

/* --- simple media query hook (client-only) --- */
function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const m = window.matchMedia(query);
    const listener = () => setMatches(m.matches);
    listener();
    m.addEventListener?.("change", listener);
    return () => m.removeEventListener?.("change", listener);
  }, [query]);
  return matches;
}

export default function ProgressiveOverloadWidget() {
  const [selectedExercise, setSelectedExercise] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"maxWeight" | "volume">("maxWeight");
  const {
    allExercises,
    loading: exercisesLoading,
    error: exercisesError,
  } = useExerciseSelection();

  // Select a random exercise on load
  useEffect(() => {
    if (!exercisesLoading && !exercisesError && allExercises.length > 0 && !selectedExercise) {
      const randomIndex = Math.floor(Math.random() * allExercises.length);
      setSelectedExercise(allExercises[randomIndex].name);
    }
  }, [allExercises, exercisesLoading, exercisesError, selectedExercise]);

  const { data, loading, error } = useProgressiveOverloadData(selectedExercise);
  const colors = useChartColors();
  const chartColors = useChartColorsArray();

  const isMobile = useMediaQuery("(max-width: 640px)");

  type ChartDataPoint =
    | (ProgressiveOverloadDataPoint & { current: number; mode: "maxWeight" })
    | (ProgressiveVolumeDataPoint & { current: number; mode: "volume" });

  const chartData = useMemo<ChartDataPoint[]>(() => {
    if (!data) return [];

    if (mode === "maxWeight") {
      return data.maxWeight.map(point => ({
        ...point,
        current: point.maxWeight,
        mode: "maxWeight" as const,
      }));
    }

    return data.volume.map(point => ({
      ...point,
      current: point.volume,
      mode: "volume" as const,
    }));
  }, [data, mode]);

  const yAxisDomain = useMemo(() => {
    if (!chartData.length) {
      return undefined;
    }

    const values: number[] = [];

    chartData.forEach(point => {
      if (typeof point.current === "number") values.push(point.current);
      if (typeof point.weekAgo === "number") values.push(point.weekAgo);
      if (typeof point.monthAgo === "number") values.push(point.monthAgo);
      if (typeof point.yearAgo === "number") values.push(point.yearAgo);
    });

    if (values.length === 0) {
      return undefined;
    }

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue;
    const padding = range === 0 ? Math.max(minValue * 0.1, 2) : range * 0.1;
    const lowerBound = Math.max(0, minValue - padding);
    const upperBound = maxValue + padding;

    return [lowerBound, upperBound] as [number, number];
  }, [chartData]);

  const yAxisLabel = mode === "maxWeight" ? "Max Weight (kg)" : "Total Volume (kg)";
  const currentLineName = mode === "maxWeight" ? "Current" : "Current Volume";
  const currentLegendLabel = mode === "maxWeight" ? "Current" : "Current Volume";
  const hasChartData = chartData.length > 0;

  const chartContainerClasses = [
    "w-full p-3 sm:p-4 border rounded-lg bg-card hover:shadow-md transition-shadow flex flex-col",
    "h-[68vw] min-h-[300px] sm:h-[40vw] sm:min-h-[280px]",
  ].join(" ");

  const chartSkeleton = (
    <div className={chartContainerClasses}>
      <div className='flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3 sm:mb-4 gap-3'>
        <div className='space-y-2'>
          <div className='h-3 w-20 rounded bg-muted/80 animate-pulse' />
          <div className='h-3 w-28 rounded bg-muted/60 animate-pulse' />
        </div>
        <div className='flex items-center gap-2 sm:gap-3'>
          <div className='h-7 w-24 rounded-full bg-muted animate-pulse' />
          <div className='h-7 w-24 rounded-full bg-muted/70 animate-pulse' />
        </div>
      </div>
      <div className='flex-1 rounded-md bg-muted animate-pulse' />
      <div className='mt-2 sm:mt-3 flex flex-wrap items-center justify-center gap-3 sm:gap-6'>
        <div className='h-3 sm:h-4 w-20 sm:w-28 rounded-full bg-muted animate-pulse' />
        <div className='h-3 sm:h-4 w-20 sm:w-28 rounded-full bg-muted/80 animate-pulse' />
        <div className='h-3 sm:h-4 w-20 sm:w-28 rounded-full bg-muted/60 animate-pulse' />
        <div className='h-3 sm:h-4 w-20 sm:w-28 rounded-full bg-muted/40 animate-pulse' />
      </div>
    </div>
  );

  if (exercisesLoading) {
    return (
      <WidgetWrapper>
        <WidgetHeader
          title='Progressive Overload'
          isAccordion
        />
        <AccordionContent>{chartSkeleton}</AccordionContent>
      </WidgetWrapper>
    );
  }

  if (exercisesError) {
    return (
      <WidgetWrapper>
        <WidgetHeader
          title='Progressive Overload'
          isAccordion
        >
          <div className='flex items-center gap-3'>
            <Button
              variant="outline"
              role="combobox"
              disabled
              className="w-80 justify-between"
            >
              Error loading exercises
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>
          </div>
        </WidgetHeader>
        <AccordionContent>
          <div className='text-destructive bg-destructive/10 p-4 rounded-lg'>
            Error loading exercises: {exercisesError}
          </div>
        </AccordionContent>
      </WidgetWrapper>
    );
  }

  // balance left/right so the plot stays visually centered
  const yAxisW = isMobile ? 36 : 48;
  const chartMargin = isMobile
    ? { top: 8, right: 12, left: 12, bottom: 20 }
    : { top: 16, right: 24, left: 24, bottom: 28 };

  return (
    <WidgetWrapper>
      <WidgetHeader
        title='Progressive Overload'
        isAccordion
      >
        <div className='flex items-center gap-3'>
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                role="combobox"
                aria-expanded={open}
                className="w-80 justify-between"
              >
                {selectedExercise
                  ? allExercises.find((exercise) => exercise.name === selectedExercise)?.name
                  : "Select an exercise..."}
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-0">
              <Command>
                <CommandInput placeholder="Search exercises..." />
                <CommandList>
                  <CommandEmpty>No exercise found.</CommandEmpty>
                  <CommandGroup>
                    {allExercises.map((exercise) => (
                      <CommandItem
                        key={exercise.name}
                        value={exercise.name}
                        onSelect={(currentValue) => {
                          setSelectedExercise(currentValue === selectedExercise ? null : currentValue);
                          setOpen(false);
                        }}
                      >
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            selectedExercise === exercise.name ? "opacity-100" : "opacity-0"
                          )}
                        />
                        {exercise.name}
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        </div>
      </WidgetHeader>

      <AccordionContent>
        {loading && chartSkeleton}

        {error && (
          <div className='text-destructive bg-destructive/10 p-4 rounded-lg'>
            Error loading data: {error}
          </div>
        )}

        {!selectedExercise && !loading && (
          <div className='h-96 flex items-center justify-center bg-muted/30 rounded-lg border-2 border-dashed border-muted-foreground/25'>
            <div className='text-center'>
              <p className='text-muted-foreground text-lg mb-2'>
                Select an exercise to view progression
              </p>
              <p className='text-sm text-muted-foreground'>
                Choose from the dropdown above to see max weight over time
              </p>
            </div>
          </div>
        )}

        {data && data.maxWeight.length === 0 && data.volume.length === 0 && selectedExercise && !loading && (
          <div className='h-96 flex items-center justify-center bg-muted/30 rounded-lg border-2 border-dashed border-muted-foreground/25'>
            <div className='text-center'>
              <p className='text-muted-foreground text-lg mb-2'>
                No data available
              </p>
              <p className='text-sm text-muted-foreground'>
                No training data found for {selectedExercise}
              </p>
            </div>
          </div>
        )}

        {selectedExercise && !loading && data && !hasChartData && (data.maxWeight.length > 0 || data.volume.length > 0) && (
          <div className='h-96 flex items-center justify-center bg-muted/30 rounded-lg border-2 border-dashed border-muted-foreground/25'>
            <div className='text-center'>
              <p className='text-muted-foreground text-lg mb-2'>
                {mode === "maxWeight" ? "No max weight data" : "No volume data"}
              </p>
              <p className='text-sm text-muted-foreground'>
                Try switching the mode or pick another exercise
              </p>
            </div>
          </div>
        )}

        {hasChartData && !loading && (
          <div className={chartContainerClasses}>
            <div className='flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3 sm:mb-4 gap-3'>
              <div>
                <p className='text-xs uppercase tracking-wide text-muted-foreground font-semibold'>Mode</p>
                <p className='text-sm text-muted-foreground'>Choose which metric to chart</p>
              </div>
              <RadioGroup
                value={mode}
                onValueChange={(value) => setMode((value as "maxWeight" | "volume"))}
                className='flex flex-wrap items-center gap-4 sm:gap-6'
                orientation='horizontal'
              >
                <div className='flex items-center gap-2'>
                  <RadioGroupItem id='mode-max-weight' value='maxWeight' />
                  <label
                    htmlFor='mode-max-weight'
                    className='text-sm font-medium leading-none text-foreground'
                  >
                    Max Weight
                  </label>
                </div>
                <div className='flex items-center gap-2'>
                  <RadioGroupItem id='mode-volume' value='volume' />
                  <label
                    htmlFor='mode-volume'
                    className='text-sm font-medium leading-none text-foreground'
                  >
                    Volume
                  </label>
                </div>
              </RadioGroup>
            </div>
            <div className='flex-1'>
              <ResponsiveContainer
                width='100%'
                height='100%'
              >
                <LineChart
                  data={chartData}
                  margin={chartMargin}
                >
                  <XAxis
                    dataKey='date'
                    tick={{
                      fontSize: isMobile ? 10 : 12,
                      fill: "currentColor",
                    }}
                    tickLine={{ stroke: "currentColor" }}
                    axisLine={{ stroke: "currentColor" }}
                    angle={isMobile ? 0 : -45}
                    textAnchor={isMobile ? "middle" : "end"}
                    height={isMobile ? 26 : 44}
                    tickMargin={isMobile ? 6 : 8}
                    interval='preserveStartEnd'
                    tickFormatter={(date) =>
                      new Date(date).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                      })
                    }
                  />

                  {/* Left axis (real) */}
                  <YAxis
                    yAxisId='L'
                    width={yAxisW}
                    tick={{
                      fontSize: isMobile ? 10 : 12,
                      fill: "currentColor",
                    }}
                    tickLine={{ stroke: "currentColor" }}
                    axisLine={{ stroke: "currentColor" }}
                    domain={yAxisDomain}
                    label={{
                      value: yAxisLabel,
                      angle: -90,
                      position: "insideLeft",
                      style: { textAnchor: "middle", fill: "currentColor" },
                    }}
                  />

                  {/* Right axis (dummy) to balance spacing */}
                  <YAxis
                    yAxisId='R'
                    orientation='right'
                    width={yAxisW}
                    tick={false}
                    axisLine={false}
                    domain={yAxisDomain}
                  />

                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length && selectedExercise) {
                        const dataPoint = payload.find(p => p.dataKey === 'current')?.payload as ChartDataPoint | undefined;

                        const isMaxWeightPoint = dataPoint?.mode === "maxWeight";
                        const isVolumePoint = dataPoint?.mode === "volume";
                        const showDistance =
                          isMaxWeightPoint && shouldDisplayDistance(selectedExercise);
                        const distanceUnit = getDistanceUnit(selectedExercise);

                        return (
                          <div className='bg-popover p-2 sm:p-3 border rounded-lg shadow-lg min-w-40 sm:min-w-48'>
                            <p className='text-xs sm:text-sm font-medium mb-1 sm:mb-2 text-popover-foreground'>
                              {label
                                ? new Date(label).toLocaleDateString("en-US", {
                                    weekday: isMobile ? undefined : "short",
                                    year: "numeric",
                                    month: "short",
                                    day: "numeric",
                                  })
                                : "Unknown date"}
                            </p>
                            {isMaxWeightPoint && dataPoint && (
                              <p className='text-xs sm:text-sm text-muted-foreground mb-1'>
                                {showDistance
                                  ? `Distance: ${dataPoint.reps ?? 0} ${distanceUnit}`
                                  : `Reps: ${dataPoint.reps ?? "-"}`}
                              </p>
                            )}
                            {isVolumePoint && dataPoint && (
                              <p className='text-xs sm:text-sm text-muted-foreground mb-1'>
                                Sets: {dataPoint.sets ?? "-"}
                              </p>
                            )}
                            {payload.map((entry, index) => (
                              <p
                                key={index}
                                className='text-xs sm:text-sm'
                                style={{ color: entry.color }}
                              >
                                {entry.name}:{" "}
                                {Math.round(entry.value as number)} kg
                              </p>
                            ))}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />

                  {/* Lines */}
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='current'
                    stroke={chartColors[0]}
                    strokeWidth={3}
                    dot={{ fill: chartColors[0], r: isMobile ? 3 : 4 }}
                    activeDot={{
                      r: isMobile ? 5 : 6,
                      stroke: chartColors[0],
                      strokeWidth: 2,
                      fill: colors.background,
                    }}
                    name={currentLineName}
                  />
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='weekAgo'
                    stroke={chartColors[1]}
                    strokeWidth={2}
                    strokeDasharray='8 4'
                    dot={{ fill: chartColors[1], r: isMobile ? 2 : 3 }}
                    connectNulls={false}
                    name={isMobile ? "1W Ago" : "1 Week Ago"}
                  />
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='monthAgo'
                    stroke={chartColors[2]}
                    strokeWidth={2}
                    strokeDasharray='12 6'
                    dot={{ fill: chartColors[2], r: isMobile ? 2 : 3 }}
                    connectNulls={false}
                    name={isMobile ? "1M Ago" : "1 Month Ago"}
                  />
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='yearAgo'
                    stroke={chartColors[3]}
                    strokeWidth={2}
                    strokeDasharray='16 8'
                    dot={{ fill: chartColors[3], r: isMobile ? 2 : 3 }}
                    connectNulls={false}
                    name={isMobile ? "1Y Ago" : "1 Year Ago"}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Compact, wrapping legend below the chart on mobile */}
            <div
              className={[
                "mt-2 sm:mt-3 flex flex-wrap items-center justify-center gap-3 sm:gap-6",
                "text-xs sm:text-sm",
              ].join(" ")}
            >
              <LegendItem
                label={isMobile ? "1M Ago" : "1 Month Ago"}
                color={chartColors[2]}
              />
              <LegendItem
                label={isMobile ? "1W Ago" : "1 Week Ago"}
                color={chartColors[1]}
              />
              <LegendItem
                label={isMobile ? "1Y Ago" : "1 Year Ago"}
                color={chartColors[3]}
              />
              <LegendItem
                label={currentLegendLabel}
                color={chartColors[0]}
                thick
              />
            </div>
          </div>
        )}
      </AccordionContent>
    </WidgetWrapper>
  );
}

/** Small helper for custom legend */
function LegendItem({
  label,
  color,
  thick = false,
}: {
  label: string;
  color: string;
  thick?: boolean;
}) {
  return (
    <div className='inline-flex items-center gap-2'>
      <span
        className='inline-block rounded-full'
        style={{
          width: 14,
          height: thick ? 4 : 3,
          background: color,
        }}
      />
      <span style={{ color }}>{label}</span>
    </div>
  );
}
