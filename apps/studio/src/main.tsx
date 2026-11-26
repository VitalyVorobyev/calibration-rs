import React from "react";
import ReactDOM from "react-dom/client";
import { initTheme, TooltipProvider } from "@vitavision/ui";
import { App } from "./App";
import { THEME_STORAGE_KEY } from "./lib/theme";
import { openServedRun } from "./store/servedRun";
import "./index.css";

// The inline script in index.html already painted the stored (or OS) theme;
// this keeps a "system" choice following the OS while the app runs.
initTheme(THEME_STORAGE_KEY);

// In the browser viewer, load the run it serves (a no-op elsewhere).
void openServedRun();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TooltipProvider>
      <App />
    </TooltipProvider>
  </React.StrictMode>,
);
