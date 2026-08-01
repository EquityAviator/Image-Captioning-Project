/**
 * useSpeechSynthesis — speak text using the browser's TTS engine.
 */

"use client";

import { useCallback, useState } from "react";
import { useSyncExternalStore } from "react";

const emptySubscribe = () => () => {};

function getSpeechSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function useSpeechSynthesis() {
  // useSyncExternalStore returns false on the server, the real value on client.
  const supported = useSyncExternalStore(
    emptySubscribe,
    getSpeechSupported,
    () => false,
  );
  const [speaking, setSpeaking] = useState(false);

  const speak = useCallback(
    (text: string) => {
      if (!supported || !text) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1;
      u.pitch = 1;
      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(u);
    },
    [supported],
  );

  const cancel = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  return { supported, speaking, speak, cancel };
}
