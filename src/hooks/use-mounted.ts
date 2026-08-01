/**
 * useMounted — true after first client render (avoids hydration mismatch).
 */

"use client";

import { useSyncExternalStore } from "react";

const emptySubscribe = () => () => {};

export function useMounted(): boolean {
  // useSyncExternalStore avoids the "setState in effect" lint entirely:
  // the server snapshot is always false, the client snapshot becomes true
  // after hydration.
  return useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );
}
