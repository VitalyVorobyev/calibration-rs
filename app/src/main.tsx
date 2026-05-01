import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/geist-mono/400.css";
import "@fontsource/geist-mono/600.css";
import { App } from "./App";
import "./index.css";

// Apply the persisted theme (or the OS preference when there is no
// override) before React mounts so the first paint is correctly
// themed and we don't flash light-on-dark / dark-on-light.
const stored = localStorage.getItem("calib-theme");
const prefersDark =
  typeof matchMedia === "function" &&
  matchMedia("(prefers-color-scheme: dark)").matches;
const dark = stored ? stored === "dark" : prefersDark;
document.documentElement.classList.toggle("dark", dark);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
