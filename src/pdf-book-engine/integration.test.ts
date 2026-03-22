import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { generateBook } from './index.js';
import { parseHtml } from './html-parser.js';
import { LULU_PROFILE, KDP_PROFILE, INGRAM_PROFILE } from './printer-profiles.js';
import type { Chapter, PdfBookConfig } from './types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fontsDir = resolve(__dirname, '../../examples/fonts');

// Check if fonts are available for integration tests
const fontsAvailable =
  existsSync(resolve(fontsDir, 'EBGaramond-Regular.ttf')) &&
  existsSync(resolve(fontsDir, 'EBGaramond-Italic.ttf')) &&
  existsSync(resolve(fontsDir, 'EBGaramond-Bold.ttf'));

function makeConfig(overrides: Partial<PdfBookConfig> = {}): PdfBookConfig {
  return {
    trimWidth: 6,
    trimHeight: 9,
    margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
    fonts: {
      body: resolve(fontsDir, 'EBGaramond-Regular.ttf'),
      bodyItalic: resolve(fontsDir, 'EBGaramond-Italic.ttf'),
      bodyBold: resolve(fontsDir, 'EBGaramond-Bold.ttf'),
      heading: resolve(fontsDir, 'EBGaramond-Bold.ttf'),
    },
    fontSize: 11,
    lineHeight: 1.4,
    paragraphIndent: 1.5,
    chapterStartRecto: true,
    chapterTopDrop: 2,
    widowLines: 2,
    orphanLines: 2,
    ...overrides,
  };
}

const sampleChapters: Chapter[] = [
  {
    title: 'First Chapter',
    html: `
      <p>This is the first paragraph of the first chapter. It contains enough
      text to ensure that line breaking and page layout are properly exercised
      during the integration test.</p>
      <p>Here is a second paragraph with <em>italic text</em> and
      <strong>bold text</strong> to verify inline style handling.</p>
    `,
  },
  {
    title: 'Second Chapter',
    html: `
      <p>The second chapter begins here. It includes various HTML elements
      to test the full pipeline.</p>
      <h2>A Section Heading</h2>
      <p>Content under the section heading with a
      <a href="https://example.com">link</a> for footnote generation.</p>
      <h3>A Sub-Section</h3>
      <p>Content under the sub-section.</p>
      <ul>
        <li>First item in an unordered list</li>
        <li>Second item with <em>emphasis</em></li>
        <li>Third item</li>
      </ul>
      <ol>
        <li>First ordered item</li>
        <li>Second ordered item</li>
      </ol>
    `,
  },
];

describe.skipIf(!fontsAvailable)('Integration: generateBook', () => {
  it('generates a valid PDF with basic chapters', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig());

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);

    // PDF magic bytes: %PDF-
    const header = String.fromCharCode(...pdf.slice(0, 5));
    expect(header).toBe('%PDF-');
  });

  it('generates PDF with headers and footers', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig({
      header: {
        outside: 'PAGE',
        inside: 'CHAPTER',
        separator: '|',
        fontSize: 9,
      },
      footer: {
        center: 'PAGE',
        fontSize: 10,
      },
    }));

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);
  });

  it('respects chapterStartRecto: false', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig({
      chapterStartRecto: false,
    }));

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);
  });

  it('works with different trim sizes', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig({
      trimWidth: 5.5,
      trimHeight: 8.5,
    }));

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);
  });

  it('works with KDP profile', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig({
      printer: KDP_PROFILE,
    }));

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);
  });

  it('works with IngramSpark profile', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig({
      printer: INGRAM_PROFILE,
    }));

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);
  });

  it('handles a single chapter', async () => {
    const pdf = await generateBook(
      [{ title: 'Solo', html: '<p>Just one chapter.</p>' }],
      makeConfig(),
    );

    expect(pdf).toBeInstanceOf(Uint8Array);
    const header = String.fromCharCode(...pdf.slice(0, 5));
    expect(header).toBe('%PDF-');
  });

  it('handles long content that spans many pages', async () => {
    const longParagraphs = Array.from({ length: 50 }, (_, i) =>
      `<p>This is paragraph number ${i + 1}. It contains enough words to make
      the line breaker work across multiple lines on the page, ensuring that
      page breaking logic is properly tested with realistic content volumes.</p>`
    ).join('\n');

    const pdf = await generateBook(
      [{ title: 'Long Chapter', html: longParagraphs }],
      makeConfig(),
    );

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(1000);
  });

  it('handles empty chapter HTML gracefully', async () => {
    const pdf = await generateBook(
      [{ title: 'Empty', html: '' }],
      makeConfig(),
    );

    expect(pdf).toBeInstanceOf(Uint8Array);
    expect(pdf.length).toBeGreaterThan(0);
  });

  it('generates different output for different font sizes', async () => {
    const pdf11 = await generateBook(sampleChapters, makeConfig({ fontSize: 11 }));
    const pdf14 = await generateBook(sampleChapters, makeConfig({ fontSize: 14 }));

    // Different font sizes should produce different PDFs
    expect(pdf11.length).not.toBe(pdf14.length);
  });

  it('works with fonts passed as ArrayBuffers', async () => {
    const pdf = await generateBook(sampleChapters, makeConfig({
      fonts: {
        body: readFileSync(resolve(fontsDir, 'EBGaramond-Regular.ttf')).buffer as ArrayBuffer,
        bodyItalic: readFileSync(resolve(fontsDir, 'EBGaramond-Italic.ttf')).buffer as ArrayBuffer,
        bodyBold: readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')).buffer as ArrayBuffer,
      },
    }));

    expect(pdf).toBeInstanceOf(Uint8Array);
    const header = String.fromCharCode(...pdf.slice(0, 5));
    expect(header).toBe('%PDF-');
  });
});
