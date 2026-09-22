import { create } from "zustand";

type Theme = "dark" | "light";

interface ThemeState {
  theme: Theme;
  toggle: () => void;
}

function readInitialTheme(): Theme {
  try {
    const saved = localStorage.getItem("sirlab-theme");
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* ignore: private browsing / blocked storage */
  }
  return "dark";
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readInitialTheme(),
  toggle: () => {
    const next = get().theme === "dark" ? "light" : "dark";
    set({ theme: next });
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("sirlab-theme", next);
    } catch {
      /* ignore */
    }
  },
}));
