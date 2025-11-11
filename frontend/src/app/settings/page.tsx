"use client";

import { useState, useEffect, useMemo } from "react";
import { useThemeColors } from "@/hooks/useThemeColors";
import { getChartColors, oklchCssToHex } from "@/lib/colors";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";
import { RotateCcw, Shuffle } from "lucide-react";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend } from "recharts";

export default function SettingsPage() {
  const {
    customChartColors,
    isLoaded,
    updateChartColor,
    setRandomChartColors,
    resetChartColors,
  } = useThemeColors();
  const [tempChartColors, setTempChartColors] = useState<string[]>(Array(5).fill(""));

  useEffect(() => {
    if (isLoaded) {
      const currentChartColors = customChartColors || getChartColors();
      const hexColors = currentChartColors.map((color) => {
        if (color.startsWith("#")) return color;
        return oklchCssToHex(color);
      });
      setTempChartColors(hexColors);
    }
  }, [isLoaded, customChartColors]);

  const handleChartColorChange = (index: number, hexColor: string) => {
    const newColors = [...tempChartColors];
    newColors[index] = hexColor;
    setTempChartColors(newColors);
    // Apply immediately if valid hex color
    if (hexColor && /^#[0-9A-F]{6}$/i.test(hexColor)) {
      updateChartColor(index, hexColor);
    }
  };

  const handleRandomColors = () => {
    setRandomChartColors();
  };

  const handleResetChartColors = () => {
    resetChartColors();
  };

  // Demo chart data
  const demoChartData = useMemo(() => {
    return Array.from({ length: 12 }, (_, i) => ({
      month: `Month ${i + 1}`,
      series1: 20 + Math.sin(i * 0.5) * 10 + Math.random() * 5,
      series2: 30 + Math.cos(i * 0.5) * 10 + Math.random() * 5,
      series3: 40 + Math.sin(i * 0.7) * 8 + Math.random() * 5,
      series4: 25 + Math.cos(i * 0.6) * 12 + Math.random() * 5,
      series5: 35 + Math.sin(i * 0.4) * 15 + Math.random() * 5,
    }));
  }, []);

  // Get current chart colors for demo - prefer tempChartColors if all valid, otherwise use saved/current colors
  const currentChartColors = useMemo(() => {
    // If all temp colors are valid hex colors, use them for immediate preview
    const allTempValid = tempChartColors.every(c => c && /^#[0-9A-F]{6}$/i.test(c));
    if (allTempValid && tempChartColors.filter(c => c).length === 5) {
      return tempChartColors;
    }
    // Otherwise use saved custom colors or current CSS colors
    const colors = customChartColors || getChartColors();
    return colors.map(color => {
      if (color.startsWith("#")) return color;
      return oklchCssToHex(color);
    });
  }, [tempChartColors, customChartColors]);

  return (
    <div className="min-h-screen bg-background">
      <div className="py-8">
        <header className="mb-8 w-full">
          <div className="flex justify-between items-center mb-4 flex-wrap gap-4">
            <div>
              <h1 className="text-3xl font-bold text-foreground mb-2">Settings</h1>
              <p className="text-muted-foreground">Customize your theme colors</p>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="outline">Back to Dashboard</Button>
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main className="space-y-6 max-w-4xl">
          <Card>
            <CardHeader>
              <CardTitle>Chart Colors</CardTitle>
              <CardDescription>
                Customize individual chart colors used in visualizations throughout the app.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <Label>Individual Chart Colors</Label>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleRandomColors}>
                    <Shuffle className="h-4 w-4 mr-2" />
                    Random Colors
                  </Button>
                  {customChartColors && (
                    <Button variant="outline" size="sm" onClick={handleResetChartColors}>
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Reset to Default
                    </Button>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[0, 1, 2, 3, 4].map((index) => (
                  <div key={index} className="space-y-2">
                    <Label htmlFor={`chart-color-${index}`}>Chart {index + 1}</Label>
                    <div className="flex gap-2 items-center">
                      <Input
                        id={`chart-color-${index}`}
                        type="color"
                        value={tempChartColors[index] || "#000000"}
                        onChange={(e) => handleChartColorChange(index, e.target.value)}
                        className="h-10 w-20 cursor-pointer flex-shrink-0"
                      />
                      <Input
                        type="text"
                        placeholder="#000000"
                        value={tempChartColors[index] || ""}
                        onChange={(e) => handleChartColorChange(index, e.target.value)}
                        pattern="^#[0-9A-F]{6}$"
                        className="font-mono flex-1"
                      />
                    </div>
                    <div
                      className="w-full h-8 rounded-md border"
                      style={{ backgroundColor: tempChartColors[index] || "#000000" }}
                    />
                  </div>
                ))}
              </div>

              {customChartColors && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Custom chart colors are active. These will be used in all chart visualizations.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Color Preview</CardTitle>
              <CardDescription>
                See how your chart colors look in a live demo chart
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={demoChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="series1"
                      stroke={currentChartColors[0] || "#3b82f6"}
                      strokeWidth={2}
                      dot={{ fill: currentChartColors[0] || "#3b82f6", r: 3 }}
                      name="Series 1"
                    />
                    <Line
                      type="monotone"
                      dataKey="series2"
                      stroke={currentChartColors[1] || "#10b981"}
                      strokeWidth={2}
                      dot={{ fill: currentChartColors[1] || "#10b981", r: 3 }}
                      name="Series 2"
                    />
                    <Line
                      type="monotone"
                      dataKey="series3"
                      stroke={currentChartColors[2] || "#8b5cf6"}
                      strokeWidth={2}
                      dot={{ fill: currentChartColors[2] || "#8b5cf6", r: 3 }}
                      name="Series 3"
                    />
                    <Line
                      type="monotone"
                      dataKey="series4"
                      stroke={currentChartColors[3] || "#f59e0b"}
                      strokeWidth={2}
                      dot={{ fill: currentChartColors[3] || "#f59e0b", r: 3 }}
                      name="Series 4"
                    />
                    <Line
                      type="monotone"
                      dataKey="series5"
                      stroke={currentChartColors[4] || "#ef4444"}
                      strokeWidth={2}
                      dot={{ fill: currentChartColors[4] || "#ef4444", r: 3 }}
                      name="Series 5"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </main>
      </div>
    </div>
  );
}

