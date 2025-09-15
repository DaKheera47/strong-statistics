"use client"

import * as React from "react"
import { Check, ChevronsUpDown, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

interface ExerciseWithLastActivity {
  name: string;
  lastActivityDate: string;
}

interface ExerciseFilterProps {
  allExercises: ExerciseWithLastActivity[];
  selectedExercises: string[];
  onSelectionChange: (selected: string[]) => void;
  loading?: boolean;
  error?: string | null;
  placeholder?: string;
  className?: string;
}

// Utility function to get recent exercises
export function getRecentExercises(allExercises: ExerciseWithLastActivity[]): string[] {
  const twoWeeksAgo = new Date();
  twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);
  const twoWeeksAgoStr = twoWeeksAgo.toISOString().split('T')[0];

  return allExercises
    .filter(exercise => exercise.lastActivityDate >= twoWeeksAgoStr)
    .map(ex => ex.name);
}

export function ExerciseFilter({
  allExercises,
  selectedExercises,
  onSelectionChange,
  loading = false,
  error = null,
  placeholder = "Select exercises...",
  className
}: ExerciseFilterProps) {
  const [open, setOpen] = React.useState(false)

  // Check if exercise is recent (within last 2 weeks)
  const isRecentExercise = (exercise: ExerciseWithLastActivity) => {
    const twoWeeksAgo = new Date();
    twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);
    const twoWeeksAgoStr = twoWeeksAgo.toISOString().split('T')[0];
    return exercise.lastActivityDate >= twoWeeksAgoStr;
  }

  const handleSelect = (exerciseName: string) => {
    if (selectedExercises.includes(exerciseName)) {
      // Prevent deselecting if it would result in less than 2 exercises
      if (selectedExercises.length > 2) {
        onSelectionChange(selectedExercises.filter((item) => item !== exerciseName))
      }
    } else {
      onSelectionChange([...selectedExercises, exerciseName])
    }
  }

  const handleClear = () => {
    // Keep the first 2 exercises when "clearing"
    const firstTwoExercises = allExercises.slice(0, 2).map(ex => ex.name);
    onSelectionChange(firstTwoExercises);
  }

  const handleSelectAll = () => {
    if (selectedExercises.length === allExercises.length) {
      onSelectionChange([])
    } else {
      onSelectionChange(allExercises.map(ex => ex.name))
    }
  }


  if (loading) {
    return (
      <div className={cn("h-10 bg-muted animate-pulse rounded-md w-[300px]", className)} />
    );
  }

  if (error) {
    return (
      <div className={cn("text-destructive text-sm", className)}>
        Error loading exercises: {error}
      </div>
    );
  }

  // Only consider exercises that are actually available in allExercises
  const availableExerciseNames = allExercises.map(ex => ex.name);
  const validSelectedExercises = selectedExercises.filter(name => availableExerciseNames.includes(name));

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn("w-[300px] justify-between", className)}
        >
          {validSelectedExercises.length === 0 
            ? placeholder
            : validSelectedExercises.length === 1 
            ? validSelectedExercises[0]
            : `${validSelectedExercises.length} exercises selected`
          }
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0">
        <Command>
          <CommandInput placeholder="Search exercises..." />
          <CommandList>
            <CommandEmpty>No exercises found.</CommandEmpty>
            <CommandGroup>
              <CommandItem onSelect={handleSelectAll}>
                <Checkbox 
                  checked={validSelectedExercises.length === allExercises.length}
                  className="mr-2"
                />
                Select All
              </CommandItem>
              {validSelectedExercises.length > 0 && (
                <CommandItem onSelect={handleClear}>
                  <X className="mr-2 h-4 w-4" />
                  Clear Selection
                </CommandItem>
              )}
              {allExercises.map((exercise) => {
                const isRecent = isRecentExercise(exercise);
                return (
                  <CommandItem
                    key={exercise.name}
                    onSelect={() => handleSelect(exercise.name)}
                    className={cn(
                      validSelectedExercises.includes(exercise.name) && validSelectedExercises.length <= 2 ?
                        "opacity-75" : ""
                    )}
                  >
                    <Checkbox 
                      checked={validSelectedExercises.includes(exercise.name)}
                      disabled={validSelectedExercises.includes(exercise.name) && validSelectedExercises.length <= 2}
                      className="mr-2"
                    />
                    <span className={cn(
                      isRecent ? "text-foreground" : "text-muted-foreground"
                    )}>
                      {exercise.name}
                    </span>
                    {!isRecent && (
                      <span className="ml-auto text-xs text-muted-foreground">
                        inactive
                      </span>
                    )}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}