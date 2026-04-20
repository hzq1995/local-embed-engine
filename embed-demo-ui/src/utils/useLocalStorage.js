import { useState } from "react";

/**
 * useState with localStorage persistence.
 * @param {string} key      - localStorage key
 * @param {*}      initial  - default value when nothing is stored
 */
export function useLocalStorage(key, initial) {
  const [value, setValueInner] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      if (stored !== null) return JSON.parse(stored);
    } catch {
      // ignore parse errors
    }
    return initial;
  });

  function setValue(next) {
    setValueInner((prev) => {
      const resolved = typeof next === "function" ? next(prev) : next;
      try {
        localStorage.setItem(key, JSON.stringify(resolved));
      } catch {
        // ignore quota errors
      }
      return resolved;
    });
  }

  return [value, setValue];
}
