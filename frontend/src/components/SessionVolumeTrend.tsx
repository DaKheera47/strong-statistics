"use client";

import { useSessionVolumeTrend } from "@/hooks/useSessionVolumeTrend";
import {
  Bar,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { useChartColors } from "@/hooks/useChartColors";
import { getChartColors } from "@/lib/colors";
import { WidgetWrapper } from "./WidgetWrapper";
import { WidgetHeader } from "./WidgetHeader";
import { AccordionContent } from "./ui/accordion";

export default function SessionVolumeTrend() {
  const { data, loading, error } = useSessionVolumeTrend();
  const colors = useChartColors();
  const chartColors = getChartColors();

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
    });
  };

  const formatTooltipDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "long",
      day: "numeric",
      year: "numeric",
    });
  };

  const formatDuration = (minutes: number | null | undefined) => {
    if (minutes === null || minutes === undefined) return null;
    if (minutes <= 0) return null;
    const hrs = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    if (hrs > 0) {
      return mins > 0 ? `${hrs}h ${mins}m` : `${hrs}h`;
    }
    return `${mins}m`;
  };

  if (loading) {
    return (
      <WidgetWrapper>
        <WidgetHeader title='Session Volume Trend' />
        <div className='h-80 bg-muted animate-pulse rounded-lg'></div>
      </WidgetWrapper>
    );
  }

  if (error) {
    return (
      <WidgetWrapper>
        <WidgetHeader title='Session Volume Trend' />
        <div className='text-destructive bg-destructive/10 p-4 rounded-lg'>
          Error loading data: {error}
        </div>
      </WidgetWrapper>
    );
  }

  if (data.length === 0) {
    return (
      <WidgetWrapper>
        <WidgetHeader title='Session Volume Trend' />
        <div className='text-muted-foreground bg-muted p-4 rounded-lg text-center'>
          No training data available.
        </div>
      </WidgetWrapper>
    );
  }

  return (
    <WidgetWrapper>
      <WidgetHeader
        title='Session Volume Trend'
        isAccordion
      />

      <AccordionContent>
        <div className='h-80 bg-card rounded-lg p-4'>
          <ResponsiveContainer
            width='100%'
            height='100%'
          >
            <ComposedChart
              data={data}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid
                strokeDasharray='3 3'
                className='opacity-20'
                vertical={false}
              />
              <XAxis
                dataKey='date'
                tickFormatter={formatDate}
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12, fill: "currentColor" }}
                interval='preserveStartEnd'
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 12, fill: "currentColor" }}
                label={{
                  value: "Volume (kg)",
                  angle: -90,
                  position: "insideLeft",
                  style: { textAnchor: "middle", fill: "currentColor" },
                }}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className='bg-popover p-3 border rounded-lg shadow-lg'>
                        <p className='text-sm font-semibold text-popover-foreground mb-2'>
                          {formatTooltipDate(label as string)}
                        </p>
                        <p className='text-sm text-popover-foreground'>
                          <span className='font-medium'>Volume:</span>{" "}
                          {Math.round(data.volume)} kg
                        </p>
                        <p className='text-xs text-muted-foreground'>
                          Sets: {data.sets}
                        </p>
                        {formatDuration(data.duration_minutes) && (
                          <p className='text-xs text-muted-foreground'>
                            Duration: {formatDuration(data.duration_minutes)}
                          </p>
                        )}
                        <p className='text-xs text-muted-foreground'>
                          Trend: {Math.round(data.trendLine)} kg (7-day avg)
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar
                dataKey='volume'
                fill={chartColors[0]}
                radius={[2, 2, 0, 0]}
                opacity={0.8}
              />
              <Line
                type='monotone'
                dataKey='trendLine'
                stroke={chartColors[1]}
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 4,
                  stroke: chartColors[1],
                  strokeWidth: 2,
                  fill: colors.background,
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </AccordionContent>
    </WidgetWrapper>
  );
}
