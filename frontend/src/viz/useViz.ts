import { useCallback, useEffect, useRef, useState } from "react";

export type Params = Record<string, string | number | undefined | null>;

export interface VizState<T> { data: T | null; error: string | null; retryable: boolean; loading: boolean; waiting: boolean; reload: () => void }

/** Payloads seen so far, by chart type, so the scorecard PDF can embed exactly what the user is looking at. */
export type Collector = (type: string, payload: unknown) => void;

/**
 * Fetches one chart's small JSON summary. `enabled=false` does nothing. A 409 (a saved run that has not finished) is retried for up to ~30 s,
 * with `waiting` set so the UI can say so. Requests are aborted when the inputs change.
 */
export function useViz<T>(datasetId: string | null, type: string, params: Params = {}, enabled = true, collect?: Collector): VizState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [retryable, setRetryable] = useState(false);   // true for network/server failures; false for "this chart does not apply to this data"
  const [tick, setTick] = useState(0);
  const collectRef = useRef(collect);
  collectRef.current = collect;
  const key = JSON.stringify(params);

  useEffect(() => {
    if (!enabled || !datasetId) { setData(null); setError(null); setLoading(false); return; }
    const ctl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    setLoading(true); setError(null); setRetryable(false);
    const q = new URLSearchParams({ type });
    Object.entries(JSON.parse(key) as Params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") q.set(k, String(v)); });
    const started = Date.now();
    const go = async () => {
      try {
        const res = await fetch(`/api/visualize/${encodeURIComponent(datasetId)}?${q}`, { signal: ctl.signal });
        if (res.status === 409 && Date.now() - started < 30_000) { setWaiting(true); timer = setTimeout(go, 1500); return; }
        setWaiting(false);
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          setRetryable(res.status >= 500 || res.status === 409);
          throw new Error(typeof j.detail === "string" ? j.detail : res.statusText);
        }
        const body = (await res.json()) as T;
        setData(body); setLoading(false);
        collectRef.current?.(type, body);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        if (e instanceof TypeError) setRetryable(true);   // fetch itself failed (network)
        setData(null); setError((e as Error).message); setLoading(false); setWaiting(false);
      }
    };
    void go();
    return () => { ctl.abort(); if (timer) clearTimeout(timer); };
  }, [datasetId, type, key, enabled, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, retryable, loading, waiting, reload };
}
