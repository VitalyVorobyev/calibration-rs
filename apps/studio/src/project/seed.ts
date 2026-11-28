import { toast } from "@vitavision/ui";
import { api } from "../api";
import type { ProjectView, SeedSource } from "../api/generated/types";
import { useProject } from "./projectStore";

/** Make `seed` the project's `[seed]` (`null` clears it) and show the
 * project as it is after. Rejects with the back end's error when the
 * project cannot take it; the caller says so. */
export async function applyProjectSeed(
  project: ProjectView,
  seed: SeedSource | null,
): Promise<void> {
  const view = await api.setProjectSeed({
    dir: project.dir,
    revision: project.revision,
    seed,
  });
  useProject.setState({ project: view, error: null });
  toast({
    title: seed ? "Seed set" : "Seed cleared",
    description: project.name,
    tone: "success",
  });
}
