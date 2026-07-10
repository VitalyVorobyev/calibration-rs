export interface SelectProps<T extends number | string> {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (next: T) => void;
  title?: string;
  /** Options are numbers by default (every current caller is a camera/pose
   * index); pass `parse={String}` for string-valued selects if one shows up. */
  parse?: (raw: string) => T;
}

/** Compact toolbar control: an uppercase mono label followed by a native
 * `<select>`. Three workspaces hand-rolled this identically (EpipolarWorkspace's
 * `Selector`, DepthWorkspace's `Selector`, Viewer3DWorkspace's
 * `RefCameraSelect`) — this is the one definition. */
export function Select<T extends number | string>({
  label,
  value,
  options,
  onChange,
  title,
  parse = (raw) => Number(raw) as T,
}: SelectProps<T>) {
  return (
    <label className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
      <span className="uppercase tracking-wider">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(parse(e.target.value))}
        title={title}
        className="h-7 rounded-md border border-border bg-surface px-1 text-foreground"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
