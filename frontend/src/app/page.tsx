"use client";

import VolumeSparklines from "@/components/VolumeSparklines";
import MaxWeightSparklines from "@/components/MaxWeightSparklines";
import RepRangeDistribution from "@/components/RepRangeDistribution";
import ProgressiveOverloadWidget from "@/components/ProgressiveOverloadWidget";
import SessionVolumeTrend from "@/components/SessionVolumeTrend";
import RecentWorkouts from "@/components/RecentWorkouts";
import { Accordion, AccordionItem } from "@/components/ui/accordion";

export default function Home() {
  return (
    <div className='min-h-screen bg-background'>
      <div className='py-8'>
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
