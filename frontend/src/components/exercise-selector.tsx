"use client"

import * as React from "react"
import { ChevronsUpDown, X } from "lucide-react"
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

interface ExerciseSelectorProps {
  value: string[];
  onValueChange: (value: string[]) => void;
  exercises: ExerciseWithLastActivity[];
}

export function ExerciseSelector({ value, onValueChange, exercises }: ExerciseSelectorProps) {
  const [open, setOpen] = React.useState(false)

  // Check if exercise is recent (within last 2 weeks)
  const isRecentExercise = (exercise: ExerciseWithLastActivity) => {
    const twoWeeksAgo = new Date();
    twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);
    const twoWeeksAgoStr = twoWeeksAgo.toISOString().split('T')[0];
    return exercise.lastActivityDate >= twoWeeksAgoStr;
  }

  const handleSelect = (exerciseName: string) => {
    if (value.includes(exerciseName)) {
      onValueChange(value.filter((item) => item !== exerciseName))
    } else {
      onValueChange([...value, exerciseName])
    }
  }

  const handleClear = () => {
    onValueChange([])
  }

  const handleSelectAll = () => {
    if (value.length === exercises.length) {
      onValueChange([])
    } else {
      onValueChange(exercises.map(ex => ex.name))
    }
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-[300px] justify-between"
        >
          {value.length === 0 
            ? "Select exercises..." 
            : value.length === 1 
            ? value[0]
            : `${value.length} exercises selected`
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
                  checked={value.length === exercises.length}
                  className="mr-2"
                />
                Select All
              </CommandItem>
              {value.length > 0 && (
                <CommandItem onSelect={handleClear}>
                  <X className="mr-2 h-4 w-4" />
                  Clear Selection
                </CommandItem>
              )}
              {exercises.map((exercise) => {
                const isRecent = isRecentExercise(exercise);
                return (
                  <CommandItem
                    key={exercise.name}
                    onSelect={() => handleSelect(exercise.name)}
                  >
                    <Checkbox 
                      checked={value.includes(exercise.name)}
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