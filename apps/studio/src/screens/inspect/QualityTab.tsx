import { Badge, Button, Callout, ErrorBox, Skeleton } from "@vitavision/ui";
import { useLocation, useNavigate } from "react-router";
import { api } from "../../api";
import type { QualityCheck, RunRef } from "../../api/generated/types";
import { useQuery } from "../../session/query";
import { parsePlace, placePath, type InspectTab } from "../../shell/screens";
import { CameraFrame } from "./CameraFrame";
import {
  CHECK_WORDS,
  checksFor,
  measurementText,
  STATUS_WORDS,
  statusTone,
  tally,
  VERDICT_WORDS,
} from "./qualityModel";

/** The tab that shows more of what a check is about. */
const SHOW: Partial<Record<QualityCheck["kind"], { tab: InspectTab; label: string }>> = {
  projection_sigma: { tab: "sigma", label: "Show projection σ" },
  feature_coverage: { tab: "residuals", label: "Show the residual map" },
  view_outliers: { tab: "residuals", label: "Show the residuals" },
  observability: { tab: "parameters", label: "Show the parameters" },
  parameter_correlation: { tab: "parameters", label: "Show the parameters" },
  parameter_conditioning: { tab: "parameters", label: "Show the parameters" },
  covariance_linearity: { tab: "parameters", label: "Show the parameters" },
};

/** The checks of the run's quality report: each with its verdict, what was
 * measured, the limit, and one sentence. */
export function QualityTab() {
  return (
    <CameraFrame title="Quality">
      {({ run, camera }) => <Body run={run} camera={camera} />}
    </CameraFrame>
  );
}

function Body({ run, camera }: { run: RunRef; camera: number }) {
  const q = useQuery("run_quality", { run }, () => api.runQuality({ run }));
  if (q.error && !q.data) {
    return q.error.kind === "io" ? (
      <Callout tone="info" title="No quality report">
        This run has none: an imported export whose quality could not be assessed.
      </Callout>
    ) : (
      <ErrorBox>
        Could not read the quality report: {q.error.message}{" "}
        <Button size="sm" onClick={q.refetch}>
          Retry
        </Button>
      </ErrorBox>
    );
  }
  if (!q.data) return <Skeleton className="h-48 w-full" />;
  const shown = checksFor(q.data.checks, camera);
  const t = tally(shown);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
      <p
        className="flex flex-wrap items-center gap-2 text-sm text-fg"
        data-testid="verdict"
      >
        Overall verdict
        <Badge tone={statusTone(q.data.verdict)}>{VERDICT_WORDS[q.data.verdict]}</Badge>
        <span className="text-xs text-fg-muted">
          the worst of the assessed checks; {t.pass} pass, {t.warn} warning, {t.fail} fail
          {t.not_applicable > 0 ? `, ${t.not_applicable} not assessed` : ""}
        </span>
      </p>
      <ul aria-label="Quality checks" className="flex flex-col divide-y divide-line">
        {shown.map((c) => (
          <Check key={`${c.kind}-${c.camera ?? "run"}`} check={c} />
        ))}
      </ul>
    </div>
  );
}

function Check({ check }: { check: QualityCheck }) {
  const navigate = useNavigate();
  const { pathname, search } = useLocation();
  const words = CHECK_WORDS[check.kind];
  const measured = measurementText(check.measurement);
  const show = SHOW[check.kind];
  return (
    <li className="flex flex-col gap-1 py-2.5" data-testid={`check-${check.kind}`}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={statusTone(check.status)}>{STATUS_WORDS[check.status]}</Badge>
        <span className="text-sm font-medium text-fg">{words.title}</span>
        {check.camera != null && (
          <span className="font-mono text-xs text-fg-muted">camera {check.camera}</span>
        )}
        {show && check.status !== "not_applicable" && (
          <Button
            size="sm"
            variant="ghost"
            className="ml-auto"
            onClick={() =>
              void navigate(placePath({ ...parsePlace(pathname, search), tab: show.tab }))
            }
          >
            {show.label}
          </Button>
        )}
      </div>
      <p className="text-xs text-fg-muted">{words.about}</p>
      {measured && <p className="font-mono text-xs text-fg">{measured}</p>}
      <p className="text-xs text-fg">{check.message}</p>
      {(check.subjects?.length ?? 0) > 0 && (
        <p className="font-mono text-xs text-fg-muted">{check.subjects!.join(", ")}</p>
      )}
    </li>
  );
}
