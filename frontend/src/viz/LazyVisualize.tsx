import { lazy, Suspense, type ComponentProps } from "react";

// The charts libraries (Recharts, React Flow) are large; they are only downloaded when a Visualize tab is opened.
const Panel = lazy(() => import("./VisualizePanel").then((m) => ({ default: m.VisualizePanel })));

export function VisualizePanel(props: ComponentProps<typeof Panel>) {
  return (
    <Suspense fallback={<div role="status" aria-live="polite" className="vcard-state" style={{ minHeight: 200 }}><div className="vskel" style={{ maxWidth: 480 }} aria-label="Loading charts" /></div>}>
      <Panel {...props} />
    </Suspense>
  );
}
