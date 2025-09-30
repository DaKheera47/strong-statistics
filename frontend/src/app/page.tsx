"use client";

import VolumeSparklines from "@/components/VolumeSparklines";
import MaxWeightSparklines from "@/components/MaxWeightSparklines";
import RepRangeDistribution from "@/components/RepRangeDistribution";
import ProgressiveOverloadWidget from "@/components/ProgressiveOverloadWidget";
import SessionVolumeTrend from "@/components/SessionVolumeTrend";
import RecentWorkouts from "@/components/RecentWorkouts";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Accordion, AccordionItem } from "@/components/ui/accordion";

export default function Home() {
  return (
    <div className='min-h-screen bg-background'>
      <div className='py-8'>
        <header className='mb-8 w-full'>
          <div className='flex justify-between items-center mb-4 flex-wrap gap-4'>
            <div>
              <h1 className='text-3xl font-bold text-foreground mb-2'>
                Strong Statistics
              </h1>
              <p className='text-muted-foreground'>
                Your training analytics dashboard
              </p>
            </div>
            <div className='flex items-center gap-4'>
              <Link href='/webui-ingest'>
                <Button variant='outline'>Ingest CSV</Button>
              </Link>
              <Link href='/workouts'>
                <Button variant='outline'>All Workouts</Button>
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main className='space-y-8'>
          <Accordion
            type='multiple'
            defaultValue={[
              "recent-workouts",
              "progressive-overload",
              "session-volume-trend",
              "volume-sparklines",
              "max-weight-sparklines",
              "rep-range-distribution",
            ]}
            className='w-full'
          >
            <AccordionItem value='recent-workouts'>
              <RecentWorkouts />
            </AccordionItem>

            <AccordionItem value='progressive-overload'>
              <ProgressiveOverloadWidget />
            </AccordionItem>

            <AccordionItem value='session-volume-trend'>
              <SessionVolumeTrend />
            </AccordionItem>

            <AccordionItem value='volume-sparklines'>
              <VolumeSparklines />
            </AccordionItem>

            <AccordionItem value='max-weight-sparklines'>
              <MaxWeightSparklines />
            </AccordionItem>

            <AccordionItem value='rep-range-distribution'>
              <RepRangeDistribution />
            </AccordionItem>
          </Accordion>
        </main>
      </div>
    </div>
  );
}
