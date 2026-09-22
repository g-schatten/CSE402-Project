import type { ReactNode } from "react";

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="page-header">
      <div className="page-header__eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      {children && <p>{children}</p>}
    </div>
  );
}

export function Card({ children }: { children: ReactNode }) {
  return <div className="card">{children}</div>;
}

export function SectionTitle({ children, sub }: { children: ReactNode; sub?: string }) {
  return (
    <>
      <div className="section-title">{children}</div>
      {sub && <div className="section-sub">{sub}</div>}
    </>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="control">
      <label>
        <span>{label}</span>
        <span className="value">{format ? format(value) : value.toFixed(3)}</span>
      </label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

export function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="stat">
      <div className="stat__label">{label}</div>
      <div className="stat__value" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend__item" key={it.label}>
          <span className="legend__swatch" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function LoadingNote({ label = "Loading precomputed results…" }: { label?: string }) {
  return <p style={{ color: "var(--text-tertiary)", fontSize: "var(--fs-sm)" }}>{label}</p>;
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <p style={{ color: "var(--bad)", fontSize: "var(--fs-sm)" }}>
      Could not load precomputed results ({message}). Run <code>make experiments</code> to generate them.
    </p>
  );
}
