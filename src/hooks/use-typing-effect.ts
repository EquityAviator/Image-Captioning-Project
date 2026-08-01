/**
 * useTypingEffect — types out a string char-by-char.
 *
 * Uses useReducer + dispatch so we never call setState synchronously
 * inside an effect body (a lint error in React 19).
 */

"use client";

import { useEffect, useReducer, useRef } from "react";

type State = { rendered: string; target: string; enabled: boolean };
type Action =
  | { type: "reset"; text: string; enabled: boolean }
  | { type: "tick" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "reset":
      return {
        target: action.text,
        enabled: action.enabled,
        rendered: action.enabled ? "" : action.text,
      };
    case "tick": {
      if (!state.enabled) return state;
      const nextLen = Math.min(state.rendered.length + 1, state.target.length);
      return { ...state, rendered: state.target.slice(0, nextLen) };
    }
  }
}

export function useTypingEffect(text: string, enabled: boolean, speed = 18): string {
  const [state, dispatch] = useReducer(reducer, {
    rendered: enabled ? "" : text,
    target: text,
    enabled,
  });

  // Keep the latest text in a ref so the interval reads fresh data without
  // needing to re-subscribe on every keystroke.
  const textRef = useRef(text);
  useEffect(() => {
    textRef.current = text;
  }, [text]);

  // Reset whenever inputs change.
  useEffect(() => {
    dispatch({ type: "reset", text, enabled });
  }, [text, enabled]);

  // Drive the typing animation.
  useEffect(() => {
    if (!enabled) return;
    const id = window.setInterval(() => {
      // Read the latest text from the ref.
      if (textRef.current.length === 0) return;
      dispatch({ type: "tick" });
    }, speed);
    return () => window.clearInterval(id);
  }, [enabled, speed, text]);

  return state.enabled ? state.rendered : state.target;
}
