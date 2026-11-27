import { cn } from "@vitavision/ui";
import { ImageOff } from "lucide-react";
import { useThumb } from "./thumbs";

/** A cell's picture: the thumbnail once loaded, a quiet placeholder while it
 * loads, and the reason when it cannot be (`title`). The cell around it
 * names the image; the picture itself is decoration. */
export function Thumb({
  path,
  max = 160,
  className,
}: {
  path: string;
  max?: 160 | 320 | 640 | 1280;
  className?: string;
}) {
  const state = useThumb(path, max);
  if (state.kind === "ready") {
    return (
      <img
        src={state.src}
        alt=""
        draggable={false}
        className={cn("size-full object-cover", className)}
      />
    );
  }
  if (state.kind === "error") {
    return (
      <span
        title={`The image does not open: ${state.message}`}
        className={cn(
          "grid size-full place-items-center bg-raised text-defect",
          className,
        )}
      >
        <ImageOff aria-hidden className="size-4" />
      </span>
    );
  }
  return (
    <span
      aria-hidden
      className={cn("block size-full animate-pulse bg-raised", className)}
    />
  );
}
