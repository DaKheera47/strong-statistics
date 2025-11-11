"use client";

import { useState, useEffect } from "react";
import { useThemeColors } from "@/hooks/useThemeColors";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";
import { RotateCcw } from "lucide-react";

export default function SettingsPage() {
  const { primaryColor, colorPalette, isLoaded, updatePrimaryColor, resetColors } = useThemeColors();
  const [tempColor, setTempColor] = useState<string>("");

  useEffect(() => {
    if (isLoaded && primaryColor) {
      setTempColor(primaryColor);
    }
  }, [isLoaded, primaryColor]);

  const handleColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newColor = e.target.value;
    setTempColor(newColor);
  };

  const handleApply = () => {
    if (tempColor && /^#[0-9A-F]{6}$/i.test(tempColor)) {
      updatePrimaryColor(tempColor);
    }
  };

  const handleReset = () => {
    resetColors();
    setTempColor("");
  };

  const colorPreviewItems = colorPalette
    ? [
        { label: "Primary", color: colorPalette.primary },
        { label: "Secondary", color: colorPalette.secondary },
        { label: "Accent", color: colorPalette.accent },
        { label: "Muted", color: colorPalette.muted },
        { label: "Chart 1", color: colorPalette.chart1 },
        { label: "Chart 2", color: colorPalette.chart2 },
        { label: "Chart 3", color: colorPalette.chart3 },
        { label: "Chart 4", color: colorPalette.chart4 },
        { label: "Chart 5", color: colorPalette.chart5 },
      ]
    : [];

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
              <CardTitle>Theme Colors</CardTitle>
              <CardDescription>
                Choose a primary color and the system will generate all related colors
                (variants, complementary colors, chart colors, etc.) automatically.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="primary-color">Primary Color</Label>
                <div className="flex gap-4 items-end">
                  <div className="flex-1">
                    <Input
                      id="primary-color"
                      type="color"
                      value={tempColor || "#000000"}
                      onChange={handleColorChange}
                      className="h-12 w-full cursor-pointer"
                    />
                  </div>
                  <div className="flex-1">
                    <Input
                      type="text"
                      placeholder="#000000"
                      value={tempColor}
                      onChange={(e) => setTempColor(e.target.value)}
                      pattern="^#[0-9A-F]{6}$"
                      className="font-mono"
                    />
                  </div>
                  <Button onClick={handleApply} disabled={!tempColor || !/^#[0-9A-F]{6}$/i.test(tempColor)}>
                    Apply
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground">
                  Enter a hex color code (e.g., #3b82f6) or use the color picker
                </p>
              </div>

              {primaryColor && (
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between mb-4">
                    <Label>Current Primary Color</Label>
                    <Button variant="outline" size="sm" onClick={handleReset}>
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Reset to Default
                    </Button>
                  </div>
                  <div
                    className="w-full h-12 rounded-md border"
                    style={{ backgroundColor: primaryColor }}
                  />
                </div>
              )}

              {colorPalette && colorPreviewItems.length > 0 && (
                <div className="pt-4 border-t">
                  <Label className="mb-4 block">Generated Color Palette</Label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {colorPreviewItems.map((item) => (
                      <div key={item.label} className="space-y-2">
                        <div className="text-sm font-medium">{item.label}</div>
                        <div
                          className="w-full h-16 rounded-md border"
                          style={{ backgroundColor: item.color }}
                        />
                        <div className="text-xs text-muted-foreground font-mono truncate">
                          {item.color}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {!primaryColor && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    No custom theme is set. The default theme colors are being used.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>About Theme Colors</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>
                  When you select a primary color, the system automatically generates:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Light and dark mode variants of the primary color</li>
                  <li>Secondary, accent, and muted colors derived from the primary</li>
                  <li>Five chart colors with complementary and analogous hues</li>
                  <li>Foreground colors with proper contrast ratios for accessibility</li>
                </ul>
                <p className="pt-2">
                  All colors are generated using the OKLCH color space for perceptually
                  uniform color mixing and better accessibility.
                </p>
                <p className="pt-2">
                  Your theme preferences are saved locally in your browser and will persist
                  across sessions.
                </p>
              </div>
            </CardContent>
          </Card>
        </main>
      </div>
    </div>
  );
}

