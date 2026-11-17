import { useEffect, useState } from "react";

/** Reactive CSS media query (SSR-safe initial value: desktop). */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/** True on phones / narrow tablets — switches shell to tab-bar mode. */
export function useIsMobile(): boolean {
  return useMediaQuery("(max-width: 860px)");
}
