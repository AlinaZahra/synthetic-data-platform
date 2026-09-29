import type { ReactNode } from "react";

/** Static grey placeholders (no animation) that hold the layout while data loads. */
export function Skeleton({ rows = 6, cols = 5 }: { rows?: number; cols?: number }) {
  return (
    <div className="skel" role="status" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, r) => (
        <div className="skel-row" key={r} style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {Array.from({ length: cols }, (_, c) => <i key={c} style={{ width: `${55 + ((r * 7 + c * 13) % 40)}%` }} />)}
        </div>
      ))}
    </div>
  );
}

interface Props {
  loading: boolean;
  error: string | null;
  hasData: boolean;
  emptyTitle?: string;
  emptyText?: string;
  onRetry?: () => void;
  skeleton?: ReactNode;
  children: ReactNode;
}

/** One place for the four states of a live preview: empty, loading, error, ready. Stale data stays visible while refreshing. */
export function PreviewFrame({ loading, error, hasData, emptyTitle = "Nothing to preview yet", emptyText, onRetry, skeleton, children }: Props) {
  if (error && !hasData) {
    return (
      <div className="state error" role="alert">
        <b>That didn’t work</b>
        <p>{error}</p>
        {onRetry && <button className="btn" onClick={onRetry}>Try again</button>}
      </div>
    );
  }
  if (!hasData) {
    return loading ? <>{skeleton ?? <Skeleton />}</> : (
      <div className="state">
        <b>{emptyTitle}</b>
        {emptyText && <p>{emptyText}</p>}
      </div>
    );
  }
  return (
    <div className={loading ? "refreshing" : undefined} aria-busy={loading}>
      {error && <div className="err" role="alert">{error} <button className="btn" onClick={onRetry}>Retry</button></div>}
      {children}
    </div>
  );
}
