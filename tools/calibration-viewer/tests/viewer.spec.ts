import { expect, test } from '@playwright/test';

test('renders 3D geometry and image overlays from a manifest', async ({ page }) => {
  await page.goto('/?manifest=/tests/fixtures/viewer_manifest.json');
  await expect(page.getByRole('heading', { name: 'synthetic_viewer' })).toBeVisible();
  await expect(page.locator('canvas')).toBeVisible();
  await expect(page.locator('.image-stage svg')).toBeVisible();

  const dataUrlLength = await page.locator('canvas').evaluate((canvas) => {
    const c = canvas as HTMLCanvasElement;
    return c.toDataURL('image/png').length;
  });
  expect(dataUrlLength).toBeGreaterThan(1500);

  const overlayBox = await page.locator('.image-stage svg').boundingBox();
  expect(overlayBox?.width).toBeGreaterThan(100);
  expect(overlayBox?.height).toBeGreaterThan(60);

  await page.getByRole('button', { name: /Target/ }).click();
  await expect(page.getByText('Worst Target Features')).toBeVisible();
});
