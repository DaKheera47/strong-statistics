"use client";

import { useChartColors } from "@/hooks/useChartColors";
import { useExerciseSelection } from "@/hooks/useExerciseSelection";
import { useProgressiveOverloadData } from "@/hooks/useProgressiveOverloadData";
import { shouldDisplayDistance, getDistanceUnit } from "@/lib/exercise-config";
import { useEffect, useState } from "react";
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

  const isMobile = useMediaQuery("(max-width: 640px)");

  if (exercisesLoading) {
    return (
      <WidgetWrapper>
        <WidgetHeader
          title='Progressive Overload'
          isAccordion
        />
        <AccordionContent>
          <div className='h-96 bg-muted animate-pulse rounded-lg' />
        </AccordionContent>
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
        {loading && <div className='h-96 bg-muted animate-pulse rounded-lg' />}

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

        {data && data.data.length === 0 && selectedExercise && !loading && (
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

        {data && data.data.length > 0 && !loading && (
          <div
            className={[
              // Taller on phones, shorter on larger screens
              "w-full p-3 sm:p-4 border rounded-lg bg-card hover:shadow-md transition-shadow flex flex-col",
              "h-[68vw] min-h-[300px] sm:h-[40vw] sm:min-h-[280px]",
            ].join(" ")}
          >
            <div className='flex-1'>
              <ResponsiveContainer
                width='100%'
                height='100%'
              >
                <LineChart
                  data={data.data}
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
                    label={{
                      value: "Max Weight (kg)",
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
                  />

                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length && selectedExercise) {
                        const showDistance = shouldDisplayDistance(selectedExercise);
                        const distanceUnit = getDistanceUnit(selectedExercise);
                        const dataPoint = payload.find(p => p.dataKey === 'maxWeight')?.payload;
                        
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
                            {dataPoint && showDistance && (
                              <p className='text-xs sm:text-sm text-muted-foreground mb-1'>
                                Distance: {dataPoint.reps} {distanceUnit}
                              </p>
                            )}
                            {dataPoint && !showDistance && (
                              <p className='text-xs sm:text-sm text-muted-foreground mb-1'>
                                Reps: {dataPoint.reps}
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
                    dataKey='maxWeight'
                    stroke={colors.primary}
                    strokeWidth={3}
                    dot={{ fill: colors.primary, r: isMobile ? 3 : 4 }}
                    activeDot={{
                      r: isMobile ? 5 : 6,
                      stroke: colors.primary,
                      strokeWidth: 2,
                      fill: colors.background,
                    }}
                    name='Current'
                  />
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='weekAgo'
                    stroke={colors.secondary}
                    strokeWidth={2}
                    strokeDasharray='8 4'
                    dot={{ fill: colors.secondary, r: isMobile ? 2 : 3 }}
                    connectNulls={false}
                    name={isMobile ? "1W Ago" : "1 Week Ago"}
                  />
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='monthAgo'
                    stroke='#8884d8'
                    strokeWidth={2}
                    strokeDasharray='12 6'
                    dot={{ fill: "#8884d8", r: isMobile ? 2 : 3 }}
                    connectNulls={false}
                    name={isMobile ? "1M Ago" : "1 Month Ago"}
                  />
                  <Line
                    yAxisId='L'
                    type='monotone'
                    dataKey='yearAgo'
                    stroke='#82ca9d'
                    strokeWidth={2}
                    strokeDasharray='16 8'
                    dot={{ fill: "#82ca9d", r: isMobile ? 2 : 3 }}
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
                color='#8884d8'
              />
              <LegendItem
                label={isMobile ? "1W Ago" : "1 Week Ago"}
                color='#34d399'
              />
              <LegendItem
                label={isMobile ? "1Y Ago" : "1 Year Ago"}
                color='#82ca9d'
              />
              <LegendItem
                label='Current'
                color={colors.primary}
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
