import { RouterProvider } from "react-router";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { router } from "./router";

export function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}
