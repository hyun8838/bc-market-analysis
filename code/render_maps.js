// Folium HTML 두 장을 OpenStreetMap 타일 로딩 후 PNG로 렌더링한다.
// 사용: node render_maps.js input.html output.png [width height]
// Playwright와 Chrome이 필요하다. BC_CHROME_BIN으로 브라우저 경로를 지정할 수 있다.

const { chromium } = require('playwright');
const { resolve } = require('path');
const { pathToFileURL } = require('url');

const [htmlPath, pngPath, widthArg = '1200', heightArg = '760'] = process.argv.slice(2);
if (!htmlPath || !pngPath) {
  throw new Error('Folium HTML 경로와 PNG 출력 경로가 필요합니다.');
}
const width = Number(widthArg);
const height = Number(heightArg);
if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
  throw new Error('width와 height는 양의 정수여야 합니다.');
}

(async () => {
  const options = { headless: true };
  if (process.env.BC_CHROME_BIN) options.executablePath = process.env.BC_CHROME_BIN;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
    await page.goto(pathToFileURL(resolve(htmlPath)).href, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelectorAll('.leaflet-tile-loaded').length >= 6,
                               undefined, { timeout: 30000 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: resolve(pngPath) });
    const loaded = await page.locator('.leaflet-tile-loaded').count();
    console.log(`${resolve(pngPath)}: OpenStreetMap 타일 ${loaded}장 로드`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
