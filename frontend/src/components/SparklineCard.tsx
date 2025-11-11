"use client";

import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { useChartColors } from "@/hooks/useChartColors";
import { getChartColors } from "@/lib/colors";
import { cn } from "@/lib/utils";

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

export function SparklineCard({
  exercise,
  data,
  latestValue,
  delta,
}: SparklineCardProps) {
  const colors = useChartColors();
  const chartColors = getChartColors();
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
              stroke={chartColors[0]}
              strokeWidth={1.5}
              dot={{ fill: chartColors[0], r: 2 }}
              activeDot={{
                r: 3,
                stroke: chartColors[0],
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