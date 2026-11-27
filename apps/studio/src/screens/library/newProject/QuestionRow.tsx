import { Button, Field, Input, NumberInput, Select } from "@vitavision/ui";
import { Check } from "lucide-react";
import { PathField } from "../../../components/PathField";
import type { Question } from "./newProjectModel";

/** One open field of the draft manifest, asked in words with the control
 * its schema calls for: a choice of its values, a path, a number or text.
 * A field only a form can answer says where. */
export function QuestionRow({
  question,
  value,
  dataRoot,
  disabled,
  onAnswer,
}: {
  question: Question;
  value: unknown;
  /** The folder a path answer is written relative to. */
  dataRoot: string;
  disabled: boolean;
  onAnswer: (value: unknown) => void;
}) {
  const { control } = question;
  const shown = typeof value === "string" ? value : "";
  return (
    <Field
      as="group"
      label={question.text}
      description={<span className="font-mono text-[11px]">{question.path}</span>}
    >
      {control.kind === "choice" && (
        <div className="flex items-center gap-2">
          <Select
            aria-label={question.text}
            className="min-w-0 flex-1"
            value={shown}
            disabled={disabled}
            placeholder="Choose…"
            onValueChange={onAnswer}
            options={control.options.map((o) => ({
              value: o.value,
              label: o.label,
              ...(o.note ? { note: o.note } : {}),
            }))}
          />
          {/* The sniffer's guess stands until someone says it is right. */}
          {shown !== "" && (
            <Button
              size="sm"
              variant="secondary"
              disabled={disabled}
              aria-label={`Confirm: ${question.text}`}
              onClick={() => onAnswer(shown)}
            >
              <Check aria-hidden className="size-3.5" />
              Confirm
            </Button>
          )}
        </div>
      )}
      {control.kind === "text" &&
        (question.path.endsWith("path") ? (
          <PathField
            aria-label={question.text}
            value={shown}
            disabled={disabled}
            kind="file"
            base={dataRoot}
            onValueChange={onAnswer}
          />
        ) : (
          <Input
            aria-label={question.text}
            value={shown}
            disabled={disabled}
            onChange={(e) => onAnswer(e.target.value)}
          />
        ))}
      {control.kind === "number" && (
        <NumberInput
          aria-label={question.text}
          value={typeof value === "number" ? value : null}
          disabled={disabled}
          onValueChange={onAnswer}
        />
      )}
      {control.kind === "form" && (
        <p className="text-xs text-fg-muted">Set in {control.where}.</p>
      )}
    </Field>
  );
}
