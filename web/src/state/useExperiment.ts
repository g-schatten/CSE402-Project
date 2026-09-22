import { useEffect, useState } from "react";

interface ExperimentState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

const cache = new Map<string, unknown>();

/** Loads a precomputed experiment artifact from /data/<id>.json (copied
 * from results/ -- see Makefile / sirlab.io.export). Cached in-memory so
 * navigating between pages that reuse the same experiment doesn't refetch. */
export function useExperiment<T = any>(experimentId: string): ExperimentState<T> {
  const [state, setState] = useState<ExperimentState<T>>(() => {
    if (cache.has(experimentId)) {
      return { data: cache.get(experimentId) as T, loading: false, error: null };
    }
    return { data: null, loading: true, error: null };
  });

  useEffect(() => {
    if (cache.has(experimentId)) return;
    let cancelled = false;
    const base = import.meta.env.BASE_URL ?? "/";
    fetch(`${base}data/${experimentId}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`${experimentId}: HTTP ${res.status}`);
        return res.json();
      })
      .then((json) => {
        cache.set(experimentId, json);
        if (!cancelled) setState({ data: json, loading: false, error: null });
      })
      .catch((err) => {
        if (!cancelled) setState({ data: null, loading: false, error: String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [experimentId]);

  return state;
}
