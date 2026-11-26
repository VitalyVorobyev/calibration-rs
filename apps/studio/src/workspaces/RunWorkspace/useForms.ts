/** The Run workspace's two forms, as the back end serves them: the dataset
 * manifest's schema (`dataset_form`) and the topology's config schema
 * (`config_form`). Fetched on mount and when the topology changes; the
 * frontend keeps no schema copy, so its forms match the library it runs. */
import type { JsonSchema } from "@vitavision/forms";
import { useEffect, useState } from "react";
import { api } from "../../api";
import type { ProblemKind } from "../../api/generated/types";

export interface Forms {
  /** The manifest schema; `null` until loaded. */
  dataset: JsonSchema | null;
  /** The config schema of the topology; `null` until loaded or when there
   * is none. */
  config: JsonSchema | null;
  /** Why the config form did not load. */
  configError: string | null;
}

interface ConfigForm {
  topology: string;
  schema: JsonSchema | null;
  error: string | null;
}

// The back end's schemas are JSON Schema; forms' `JsonSchema` interface is
// intentionally loose, so the cast goes through `unknown`.
const asSchema = (schema: unknown) => schema as JsonSchema;

export function useForms(topology: string): Forms {
  const [dataset, setDataset] = useState<JsonSchema | null>(null);
  const [config, setConfig] = useState<ConfigForm | null>(null);

  useEffect(() => {
    let live = true;
    api.datasetForm().then(
      (form) => {
        if (live) setDataset(asSchema(form.schema));
      },
      () => {},
    );
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    let live = true;
    api.configForm({ topology: topology as ProblemKind }).then(
      (form) => {
        if (live) setConfig({ topology, schema: asSchema(form.schema), error: null });
      },
      (e: unknown) => {
        if (live) setConfig({ topology, schema: null, error: String(e) });
      },
    );
    return () => {
      live = false;
    };
  }, [topology]);

  const current = config?.topology === topology ? config : null;
  return {
    dataset,
    config: current?.schema ?? null,
    configError: current?.error ?? null,
  };
}
