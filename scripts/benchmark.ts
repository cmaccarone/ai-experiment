import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { FontManager } from '../src/pdf-book-engine/font-manager.js';
import { PdfWriter } from '../src/pdf-book-engine/pdf-writer.js';
import { PdfRenderer } from '../src/pdf-book-engine/pdf-renderer.js';
import { layoutBook } from '../src/pdf-book-engine/layout-engine.js';
import type { PdfBookConfig, FontStyle } from '../src/pdf-book-engine/types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fontsDir = resolve(__dirname, '../examples/fonts');

const fonts = {
  body: readFileSync(resolve(fontsDir, 'EBGaramond-Regular.ttf')),
  bodyItalic: readFileSync(resolve(fontsDir, 'EBGaramond-Italic.ttf')),
  bodyBold: readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')),
  heading: readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')),
};

const config: PdfBookConfig = {
  trimWidth: 6, trimHeight: 9,
  margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
  fonts, fontSize: 11, lineHeight: 1.4, paragraphIndent: 1.5,
  chapterStartRecto: true, chapterTopDrop: 2, widowLines: 2, orphanLines: 2,
};

async function run() {
  const chapters = [{ title: 'Solo', html: '<p>Just one chapter.</p>' }];

  // Step 1: Load fonts
  let t = Date.now();
  const fontManager = new FontManager();
  const fontEntries: Array<[FontStyle, Buffer]> = [
    ['body', fonts.body as Buffer],
    ['bodyItalic', fonts.bodyItalic as Buffer],
    ['bodyBold', fonts.bodyBold as Buffer],
    ['heading', fonts.heading as Buffer],
  ];
  for (const [style, source] of fontEntries) {
    await fontManager.loadFont(style, source);
  }
  console.log('Load fonts:', Date.now() - t, 'ms');

  // Step 2: Layout
  t = Date.now();
  const { pages } = await layoutBook(chapters, config, {}, fontManager);
  console.log('Layout (' + pages.length + ' pages):', Date.now() - t, 'ms');

  // Step 3: Create PDF writer + embed fonts
  t = Date.now();
  const writer = await PdfWriter.create();
  await writer.embedFonts(fontManager);
  console.log('Create writer + embed:', Date.now() - t, 'ms');

  // Step 4: Render pages
  t = Date.now();
  const renderer = new PdfRenderer(writer, config, new Map());
  for (const page of pages) {
    renderer.renderPage(page, 'Test', pages.length);
  }
  console.log('Render pages:', Date.now() - t, 'ms');

  // Step 5: Save
  t = Date.now();
  const pdf = await writer.save();
  console.log('Save (' + pdf.length + ' bytes):', Date.now() - t, 'ms');
}
run();

