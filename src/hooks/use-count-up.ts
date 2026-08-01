/**
 * useCountUp — animates a number from 0 to target on mount.
 */

"use client";

import { useEffect, useRef, useState } from "react";

export function useCountUp(target: number, durationMs = 1500, decimals = 0): number {
  const [val, setVal] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const start = performance.now();
    const animate = (t: number) => {
      const p = Math.min(1, (t - start) / durationMs);
      const eased = 1 - Math.pow(1 - p, 3); // ease-out-cubic
      setVal(target * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [target, durationMs]);

  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}
