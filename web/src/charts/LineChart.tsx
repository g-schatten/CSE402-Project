import { useMemo, useRef, useState } from "react";
import { scaleLinear, scaleLog } from "d3-scale";
import { line as d3line, curveMonotoneX } from "d3-shape";
import { extent } from "d3-array";

export interface Series {
  id: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
  width?: number;
  /** Render as a scatter of unconnected points instead of a path -- use for
   * sample clouds (bootstrap/MCMC draws) where the points have no natural
   * order and connecting them with a line would draw a meaningless scribble
   * instead of a point cloud. */
  scatter?: boolean;
  pointRadius?: number;
  opacity?: number;
}

interface LineChartProps {
  series: Series[];
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xLog?: boolean;
  yLog?: boolean;
  xDomain?: [number, number];
  yDomain?: [number, number];
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  markers?: { x: number; y: number; color: string; label?: string }[];
}

const MARGIN = { top: 12, right: 20, bottom: 36, left: 56 };

/** A hand-rolled SVG line chart (d3 for scales/shapes only, React for the
 * DOM -- PLAN.md 6.2). Handles log axes, multiple series, hover crosshair,
 * and optional marker points (used for analytic peak/final-size ghosts). */
export function LineChart({
  series,
  width = 640,
  height = 340,
  xLabel,
  yLabel,
  xLog = false,
  yLog = false,
  xDomain,
  yDomain,
  formatX = (v) => v.toFixed(2),
  formatY = (v) => v.toFixed(2),
  markers = [],
}: LineChartProps) {
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const [hover, setHover] = useState<{ x: number; values: { id: string; y: number; color: string }[] } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  const allY = series.flatMap((s) => s.points.map((p) => p.y));
  const [xMin, xMax] = xDomain ?? (extent(allX) as [number, number]) ?? [0, 1];
  const [yMinRaw, yMaxRaw] = yDomain ?? (extent(allY) as [number, number]) ?? [0, 1];
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const yMin = yLog ? Math.max(yMinRaw * 0.9, 1e-300) : yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;

  const xScale = useMemo(
    () => (xLog ? scaleLog().domain([Math.max(xMin, 1e-9), xMax]) : scaleLinear().domain([xMin, xMax])).range([0, innerW]),
    [xMin, xMax, xLog, innerW]
  );
  const yScale = useMemo(
    () => (yLog ? scaleLog().domain([yMin, yMax]) : scaleLinear().domain([yMin, yMax])).range([innerH, 0]),
    [yMin, yMax, yLog, innerH]
  );

  const lineGen = d3line<{ x: number; y: number }>()
    .x((d) => xScale(d.x))
    .y((d) => yScale(Math.max(d.y, yLog ? 1e-300 : -Infinity)))
    .curve(curveMonotoneX);

  // For a log axis, d3's default ticks() emits every "nice" 1/2/3/5x10^k
  // step within the domain, which crowds and overlaps once the domain spans
  // more than ~2 decades. We instead take one tick per decade (powers of
  // ten only) whenever the span is wide, falling back to d3's default for
  // a narrower span where the finer ticks stay legible.
  function niceTicks(scale: { domain: () => number[]; ticks: (n: number) => number[] }, isLog: boolean, count: number): number[] {
    if (!isLog) return scale.ticks(count);
    // Always use power-of-ten ticks for a log axis rather than d3's default
    // (which, for a span of a few decades, often returns an uneven mix like
    // 2e-6, 4e-6, 6e-6, 1e-5, ... that crowds and misaligns) -- one label
    // per decade, or every Nth decade if the span is very wide.
    const [lo, hi] = scale.domain();
    const startExp = Math.floor(Math.log10(lo));
    const endExp = Math.ceil(Math.log10(hi));
    const nDecades = endExp - startExp + 1;
    const step = Math.max(1, Math.ceil(nDecades / Math.max(count, 3)));
    const out: number[] = [];
    for (let e = startExp; e <= endExp; e += step) out.push(10 ** e);
    return out;
  }

  const xTicks = niceTicks(xScale, xLog, 5);
  const yTicks = niceTicks(yScale, yLog, 5);

  function handleMove(e: React.MouseEvent<SVGRectElement>) {
    const rect = (e.target as SVGRectElement).getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const xVal = xScale.invert(mx);
    const values = series.map((s) => {
      let nearest = s.points[0];
      let best = Infinity;
      for (const p of s.points) {
        const d = Math.abs(p.x - xVal);
        if (d < best) {
          best = d;
          nearest = p;
        }
      }
      return { id: s.id, y: nearest.y, color: s.color };
    });
    setHover({ x: xVal, values });
  }

  return (
    <svg ref={svgRef} width={width} height={height} role="img" aria-label={xLabel && yLabel ? `${yLabel} vs ${xLabel}` : "chart"}>
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={innerW} y1={yScale(t)} y2={yScale(t)} stroke="var(--border)" strokeWidth={1} />
            <text x={-8} y={yScale(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-secondary)" className="mono">
              {formatY(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <text x={xScale(t)} y={innerH + 20} textAnchor="middle" fontSize={11} fill="var(--text-secondary)" className="mono">
              {formatX(t)}
            </text>
          </g>
        ))}
        {series.map((s) =>
          s.scatter ? (
            <g key={s.id} opacity={s.opacity ?? 0.55}>
              {s.points.map((p, i) => (
                <circle
                  key={i}
                  cx={xScale(p.x)}
                  cy={yScale(Math.max(p.y, yLog ? 1e-300 : -Infinity))}
                  r={s.pointRadius ?? 2.5}
                  fill={s.color}
                  stroke="none"
                />
              ))}
            </g>
          ) : (
            <path
              key={s.id}
              d={lineGen(s.points) ?? undefined}
              fill="none"
              stroke={s.color}
              strokeWidth={s.width ?? 2}
              strokeDasharray={s.dashed ? "5 4" : undefined}
            />
          )
        )}
        {markers.map((m, i) => (
          <g key={i}>
            <circle cx={xScale(m.x)} cy={yScale(Math.max(m.y, yLog ? 1e-300 : -Infinity))} r={4} fill="none" stroke={m.color} strokeWidth={2} />
          </g>
        ))}
        {hover && (
          <line x1={xScale(hover.x)} x2={xScale(hover.x)} y1={0} y2={innerH} stroke="var(--text-tertiary)" strokeDasharray="3 3" />
        )}
        <rect
          x={0}
          y={0}
          width={innerW}
          height={innerH}
          fill="transparent"
          onMouseMove={handleMove}
          onMouseLeave={() => setHover(null)}
        />
        {xLabel && (
          <text x={innerW / 2} y={innerH + 34} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            x={-innerH / 2}
            y={-42}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-secondary)"
            transform={`rotate(-90)`}
          >
            {yLabel}
          </text>
        )}
      </g>
      {hover && (
        <g transform={`translate(${Math.min(width - 140, MARGIN.left + xScale(hover.x) + 10)}, ${MARGIN.top + 4})`}>
          <rect width={130} height={16 + hover.values.length * 16} rx={6} fill="var(--bg-overlay)" stroke="var(--border-strong)" />
          <text x={8} y={14} fontSize={10} fill="var(--text-secondary)" className="mono">
            {formatX(hover.x)}
          </text>
          {hover.values.map((v, i) => (
            <text key={v.id} x={8} y={30 + i * 16} fontSize={11} fill={v.color} className="mono">
              {v.id}: {formatY(v.y)}
            </text>
          ))}
        </g>
      )}
    </svg>
  );
}
