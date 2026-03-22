import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { generateBook } from '../src/pdf-book-engine/index.js';
import type { PdfBookConfig } from '../src/pdf-book-engine/types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fontsDir = resolve(__dirname, '../examples/fonts');

const config: PdfBookConfig = {
  trimWidth: 6, trimHeight: 9,
  margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
  fonts: {
    body: resolve(fontsDir, 'EBGaramond-Regular.ttf'),
    bodyItalic: resolve(fontsDir, 'EBGaramond-Italic.ttf'),
    bodyBold: resolve(fontsDir, 'EBGaramond-Bold.ttf'),
    heading: resolve(fontsDir, 'EBGaramond-Bold.ttf'),
  },
  fontSize: 11, lineHeight: 1.4, paragraphIndent: 1.5,
  chapterStartRecto: true, chapterTopDrop: 2, widowLines: 2, orphanLines: 2,
};

async function run() {
  const chapters = [{ title: 'Solo', html: '<p>Just one chapter.</p>' }];

  let t = Date.now();
  const pdf = await generateBook(chapters, config);
  console.log(`Single chapter (${pdf.length} bytes):`, Date.now() - t, 'ms');

  // Multi-chapter test
  const longChapters = Array.from({ length: 10 }, (_, i) => ({
    title: `Chapter ${i + 1}`,
    html: Array.from({ length: 20 }, (_, j) =>
      `<p>This is paragraph ${j + 1} of chapter ${i + 1}. It contains enough words to fill multiple lines and exercise the typesetting engine properly.</p>`
    ).join('\n'),
  }));

  t = Date.now();
  const longPdf = await generateBook(longChapters, config);
  console.log(`10 chapters, 200 paragraphs (${longPdf.length} bytes):`, Date.now() - t, 'ms');
}
run();
