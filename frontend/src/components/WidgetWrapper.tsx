interface WidgetWrapperProps {
  children: React.ReactNode;
}

export function WidgetWrapper({ children }: WidgetWrapperProps) {
  return (
    <div className="py-2">
      {children}
    </div>
  );
}