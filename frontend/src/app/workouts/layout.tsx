import type { Metadata } from "next";
import React from "react";

type Props = {
  children: React.ReactNode;
};

export const metadata: Metadata = {
  title: "All Workouts - Training History",
  description:
    "View and analyze your complete workout history. Browse all training sessions with detailed metrics including volume, duration, and exercise breakdown.",
  keywords: [
    "workout history",
    "training log",
    "exercise tracker",
    "session analysis",
    "fitness progress",
  ],
  openGraph: {
    title: "All Workouts - Training History | Strong Statistics",
    description:
      "View and analyze your complete workout history with detailed metrics and insights.",
  },
};

export default function Layout({ children }: Props) {
  return <>{children}</>;
}
