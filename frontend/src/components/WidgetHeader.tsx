import { AccordionTrigger } from "./ui/accordion";

interface WidgetHeaderProps {
  title: string;
  isAccordion?: boolean;
  children?: React.ReactNode;
}

export function WidgetHeader({ title, isAccordion = false, children }: WidgetHeaderProps) {
  return (
    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center md:gap-4 max-md:mb-6">
      {isAccordion ? (
        <AccordionTrigger>
          <h2 className="text-xl font-semibold text-foreground">{title}</h2>
        </AccordionTrigger>
      ) : (
        <h2 className="text-xl font-semibold text-foreground">{title}</h2>
      )}
      {children}
    </div>
  );
}