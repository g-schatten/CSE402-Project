import { useEffect, useRef } from "react";
import katex from "katex";

export function Katex({ tex, block = false }: { tex: string; block?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(tex, ref.current, { throwOnError: false, displayMode: block });
      } catch {
        if (ref.current) ref.current.textContent = tex;
      }
    }
  }, [tex, block]);
  return <span ref={ref} />;
}
