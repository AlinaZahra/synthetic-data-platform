import { useEffect, useState } from "react";
import { AssistantWorkspace } from "./AssistantView";
import { ContractsWorkspace } from "./ContractsView";
import { ExportWorkspace } from "./ExportView";
import { GuidedFlow } from "./GuidedFlow";
import { HistoryWorkspace } from "./HistoryView";
import { LocaleLab } from "./LocaleLab";
import { DocumentsWorkspace } from "./DocumentsView";
import { SchemaWorkspace } from "./SchemaView";
import { RelationalWorkspace } from "./RelationalView";
import { TabularWorkspace } from "./TabularView";

const NAV = ["Guided", "Assistant", "Tabular", "Relational", "Documents", "Schema & text", "Export & load", "Contracts", "Locale lab", "History"] as const;
type Theme = "system" | "light" | "dark";

function storedTheme(): Theme {
  try { const t = localStorage.getItem("sdp-theme"); return t === "light" || t === "dark" ? t : "system"; } catch { return "system"; }
}

export default function App() {
  const [view, setView] = useState<(typeof NAV)[number]>("Guided");
  const [theme, setTheme] = useState<Theme>(storedTheme);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", theme);
    try { if (theme === "system") localStorage.removeItem("sdp-theme"); else localStorage.setItem("sdp-theme", theme); } catch { /* storage can be unavailable */ }
  }, [theme]);

  // move keyboard focus to the new page's <main> so screen-reader and keyboard users land on the content
  useEffect(() => {
    const m = document.querySelector("main");
    if (m) { m.setAttribute("tabindex", "-1"); }
  }, [view]);

  const onNavKey = (e: React.KeyboardEvent<HTMLElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const btns = Array.from(e.currentTarget.querySelectorAll<HTMLButtonElement>("button[data-nav]"));
    const i = btns.indexOf(document.activeElement as HTMLButtonElement);
    if (i < 0) return;
    e.preventDefault();
    btns[(i + (e.key === "ArrowDown" ? 1 : -1) + btns.length) % btns.length].focus();
  };
  const next: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

  return (
    <div className="app">
      <a className="skip" href="#content" onClick={(e) => { e.preventDefault(); (document.querySelector("main") as HTMLElement | null)?.focus(); }}>Skip to content</a>
      <nav className="nav" aria-label="Engines" onKeyDown={onNavKey}>
        <div className="brand">Synthetic Data<small>Workspace · all data is fictional</small></div>
        {NAV.map((n) => (
          <button key={n} data-nav aria-current={n === view ? "page" : undefined} onClick={() => setView(n)}>{n}</button>
        ))}
        <div className="theme">
          <button onClick={() => setTheme(next[theme])} aria-label={`Colour theme: ${theme}. Activate to switch to ${next[theme]}.`}>Theme: {theme}</button>
        </div>
      </nav>
      <span id="content" className="sr-only" aria-live="polite">{view}</span>
      {view === "Guided" && <GuidedFlow />}
      {view === "Tabular" && <TabularWorkspace />}
      {view === "Relational" && <RelationalWorkspace />}
      {view === "Assistant" && <AssistantWorkspace />}
      {view === "Documents" && <DocumentsWorkspace />}
      {view === "Schema & text" && <SchemaWorkspace />}
      {view === "Export & load" && <ExportWorkspace />}
      {view === "Contracts" && <ContractsWorkspace />}
      {view === "Locale lab" && <LocaleLab />}
      {view === "History" && <HistoryWorkspace />}
    </div>
  );
}
