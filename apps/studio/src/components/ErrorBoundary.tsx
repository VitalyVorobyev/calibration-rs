import { Button, Callout } from "@vitavision/ui";
import { Component, type ErrorInfo, type ReactNode } from "react";
import { useRouteError } from "react-router";

/** The full-window fallback shown for an uncaught error. */
function ErrorFallback({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <main className="grid min-h-screen place-items-center bg-canvas p-6 text-fg">
      <div className="w-full max-w-lg space-y-3 rounded-panel border border-line bg-surface p-5">
        <Callout tone="error" title="Something went wrong">
          <p className="font-mono text-xs break-words">{message}</p>
        </Callout>
        <div className="flex justify-end">
          <Button size="sm" variant="secondary" onClick={onRetry}>
            Retry
          </Button>
        </div>
      </div>
    </main>
  );
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** The root route's `errorElement`: the data router catches errors thrown
 * while rendering a workspace before any boundary above it sees them. Retry
 * reloads the window, which re-enters the route. */
export function RouteErrorFallback() {
  const error = useRouteError();
  return (
    <ErrorFallback message={messageOf(error)} onRetry={() => window.location.reload()} />
  );
}

/** A screen's `errorElement`: the shell around it stays, the screen says it
 * failed, and the details can be reloaded or copied for a report. */
export function ScreenErrorFallback() {
  const error = useRouteError();
  const details = error instanceof Error ? (error.stack ?? error.message) : String(error);
  return (
    <div className="grid h-full place-items-center p-6">
      <div className="w-full max-w-lg space-y-3">
        <Callout tone="error" title="This screen failed">
          <p className="font-mono text-xs break-words">{messageOf(error)}</p>
        </Callout>
        <div className="flex justify-end gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void navigator.clipboard?.writeText(details)}
          >
            Copy details
          </Button>
          <Button size="sm" variant="secondary" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </div>
      </div>
    </div>
  );
}

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/** Outermost boundary, for errors outside the router (its own setup);
 * workspace errors land in `RouteErrorFallback`. */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  override state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Uncaught application error", error, info.componentStack);
  }

  override render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <ErrorFallback
        message={this.state.error.message}
        onRetry={() => this.setState({ error: null })}
      />
    );
  }
}
