import { ResidualViewer } from "./components/ResidualViewer";

export function App() {
  return (
    <main
      style={{
        height: "100%",
        width: "100%",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: "12px",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "18px", fontWeight: 600 }}>
          calibration-rs · diagnose
        </h1>
        <span style={{ opacity: 0.6, fontSize: "12px" }}>
          v0 — passive residual viewer (ADR 0014)
        </span>
      </header>
      <ResidualViewer />
    </main>
  );
}
