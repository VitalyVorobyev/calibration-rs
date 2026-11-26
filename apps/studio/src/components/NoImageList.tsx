import { Empty } from "@vitavision/ui";

/** An image pane's state for an export without an image list
 * (`diagnostics.image_manifest`): there is no image to show, and what is
 * still there to look at. */
export function NoImageList({ className }: { className?: string }) {
  return (
    <Empty className={`m-auto justify-center px-6 [&>p]:max-w-md ${className ?? ""}`}>
      This export has no image list (<code>diagnostics.image_manifest</code>), so there
      are no images to show. Residual statistics, uncertainty and the 3D view still work.
    </Empty>
  );
}
