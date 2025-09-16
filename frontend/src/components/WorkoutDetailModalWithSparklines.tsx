"use client";

import { useEffect, useState } from "react";
import WorkoutDetailModal, { WorkoutDetailData } from "./WorkoutDetailModal";

interface VolumeSparklineData {
  exercise: string;
  date: string;
  volume: number;
  sets: number;
}

interface WorkoutDetailModalWithSparklinesProps {
  isOpen: boolean;
  onClose: (open: boolean) => void;
  workout: WorkoutDetailData | null;
  isLoading: boolean;
}

export default function WorkoutDetailModalWithSparklines({
  isOpen,
  onClose,
  workout,
  isLoading,
}: WorkoutDetailModalWithSparklinesProps) {
  const [sparklineData, setSparklineData] = useState<VolumeSparklineData[]>([]);

  useEffect(() => {
    async function fetchSparklineData() {
      try {
        const response = await fetch("/api/volume-sparklines");
        if (!response.ok) {
          throw new Error("Failed to fetch volume sparklines");
        }
        const data: VolumeSparklineData[] = await response.json();
        setSparklineData(data);
      } catch (error) {
        console.error("Error fetching volume sparklines:", error);
      }
    }

    fetchSparklineData();
  }, []);

  return (
    <WorkoutDetailModal
      isOpen={isOpen}
      onClose={onClose}
      workout={workout}
      isLoading={isLoading}
      sparklineData={sparklineData}
    />
  );
}
