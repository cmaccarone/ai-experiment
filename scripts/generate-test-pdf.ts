import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { generateBook } from '../src/pdf-book-engine/index.js';
import type { Chapter, PdfBookConfig } from '../src/pdf-book-engine/types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fontsDir = resolve(__dirname, '../examples/fonts');

const fonts = {
  body: readFileSync(resolve(fontsDir, 'EBGaramond-Regular.ttf')),
  bodyItalic: readFileSync(resolve(fontsDir, 'EBGaramond-Italic.ttf')),
  bodyBold: readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')),
  heading: readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')),
};

const chapters: Chapter[] = [
  {
    title: 'Test Chapter',
    html: `
      <p>Life in Kenosha was busy. The family lived in what locals still remembered
      years later as the Maccaroni house on Old Pearl Street, now 55th Ave. Family
      members later visited the house and were told by old-timers that it was still
      known by that name.</p>
      <p>She said <em>this is very important</em>, and then she left the room quietly.
      The <strong>bold statement</strong> made everyone pause. A link to
      <a href="https://example.com">Example Site</a> was shared.</p>
      <p>Tony and <em>Lena</em> married. Their children <em>Josie</em> and <em>Sam</em>
      were born there. Family members worked nearby at the <em>Nash Motor Company</em>.
      The factory was close enough that they could walk to it.</p>
      <p>First, <em>Salvatore</em> loved the land. In <em>Italy</em> he had worked as
      a vine dresser, orchard worker, and winemaker for wealthy landowners. It is easy
      to imagine that deep down he always longed to own land of his own.</p>
      <p>Second, a friend from Italy, <em>Royal Spanish</em>, wrote to him from Walla
      Walla, telling him there were farms for sale there and urging him to come west.</p>
      <p>This is a plain body text paragraph with no italic or bold words at all. It should
      have perfectly even spacing between every single word on every justified line. If the
      spacing is uneven here then the measurement system itself has a fundamental problem
      that needs to be fixed before anything else. This paragraph is intentionally long
      enough to span multiple lines so we can see the justification behavior.</p>
    `,
  },
];

const config: PdfBookConfig = {
  trimWidth: 6,
  trimHeight: 9,
  margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
  fonts,
  fontSize: 11,
  lineHeight: 1.4,
  paragraphIndent: 1.5,
  chapterStartRecto: true,
  chapterTopDrop: 2,
  widowLines: 2,
  orphanLines: 2,
};

async function main() {
  const pdf = await generateBook(chapters, config);
  const outPath = resolve(__dirname, '../test-output.pdf');
  writeFileSync(outPath, pdf);
  console.log(`Written to ${outPath} (${pdf.length} bytes)`);
}

main().catch(console.error);
