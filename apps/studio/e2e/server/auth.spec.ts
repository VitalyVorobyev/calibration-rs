import { expect, test } from "@playwright/test";
import { E2E_PORT, E2E_TOKEN, openLoggedIn, trackErrors } from "./support";

/** The server wants its token: nothing but the page, health and login
 * answer without it; the page asks for it; the printed address (the token in
 * the fragment) logs in and leaves no token in the address bar. */

test("the API answers 401 without the token", async ({ request }) => {
  const origin = `http://127.0.0.1:${E2E_PORT}`;
  const command = await request.post("/api/cmd/server_info", {
    data: {},
    headers: { Origin: origin },
  });
  expect(command.status()).toBe(401);
  expect(await command.json()).toMatchObject({ kind: "unauthorized" });
  for (const path of ["/api/events", "/api/r/image?q=%7B%7D"]) {
    expect((await request.get(path)).status(), path).toBe(401);
  }
  expect((await request.get("/api/health")).status()).toBe(200);
  // A wrong token: still 401.
  const wrong = await request.post("/api/cmd/server_info", {
    data: {},
    headers: { Origin: origin, Authorization: "Bearer not-the-token-at-all" },
  });
  expect(wrong.status()).toBe(401);
});

test("the page asks for the token, and logs in with it", async ({ page }) => {
  await page.goto("/");
  const field = page.getByLabel("Token");
  await expect(field).toBeVisible();
  await field.fill("not-the-token-at-all");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText("That is not this server's token.")).toBeVisible();
  await field.fill(E2E_TOKEN);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();
  // The session survives a reload (the cookie), with no token asked.
  await page.reload();
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();
  await expect(page.getByLabel("Token")).toHaveCount(0);
});

test("the printed address logs in and drops the token from the URL", async ({ page }) => {
  const errors = trackErrors(page);
  await openLoggedIn(page);
  await expect(page).toHaveURL(/#\/library$/);
  expect(page.url()).not.toContain(E2E_TOKEN);
  // The cookie is the session's, not the token, and scripts cannot read it.
  const cookies = await page.context().cookies();
  const session = cookies.find((c) => c.name === "studio_session");
  expect(session?.httpOnly).toBe(true);
  expect(session?.sameSite).toBe("Strict");
  expect(session?.value).not.toBe(E2E_TOKEN);
  expect(await page.evaluate(() => document.cookie)).toBe("");
  expect(errors).toEqual([]);
});
