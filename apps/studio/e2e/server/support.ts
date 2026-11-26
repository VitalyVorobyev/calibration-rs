/** What the specs against the real server share: its port and token, a
 * logged-in page, the API with the bearer token, and console tracking. */
import { expect, type APIRequestContext, type Page } from "@playwright/test";

export const E2E_PORT = 1431;
export const E2E_TOKEN = "e2e-token-0123456789abcdefghij";

/** Errors the page logs or throws. */
export function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  return errors;
}

/** Open the app logged in, as the address the server prints does. */
export async function openLoggedIn(page: Page, extra = ""): Promise<void> {
  await page.goto(`/#token=${E2E_TOKEN}${extra}`);
  await expect(page.getByRole("heading", { name: "calibration-rs" })).toBeVisible();
  await expect(page.getByLabel("Token")).toHaveCount(0);
}

/** Run command `name` with the bearer token. */
export async function command(
  request: APIRequestContext,
  name: string,
  args: unknown,
): Promise<{ status: number; body: Record<string, unknown> }> {
  const response = await request.post(`/api/cmd/${name}`, {
    data: args,
    headers: {
      Authorization: `Bearer ${E2E_TOKEN}`,
      Origin: `http://127.0.0.1:${E2E_PORT}`,
    },
  });
  return {
    status: response.status(),
    body: (await response.json()) as Record<string, unknown>,
  };
}

/** The server's one root folder (the fixture library's copy). */
export async function libraryRoot(request: APIRequestContext): Promise<string> {
  const { body } = await command(request, "server_info", {});
  const roots = body.roots as { path: string; samples: boolean }[];
  const root = roots.find((r) => !r.samples);
  if (!root) throw new Error("the e2e server has no library root");
  return root.path;
}

/** The decoded width of the `index`-th image under `scope`. */
export async function imageWidth(page: Page, scope: string, index = 0): Promise<number> {
  const image = page.locator(`${scope} img`).nth(index);
  await expect(image).toBeAttached();
  await expect
    .poll(() => image.evaluate((img: HTMLImageElement) => img.naturalWidth))
    .toBeGreaterThan(0);
  return image.evaluate((img: HTMLImageElement) => img.naturalWidth);
}
