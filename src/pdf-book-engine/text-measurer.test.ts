import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { TextMeasurer } from './text-measurer.js';
import { FontManager } from './font-manager.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fontsDir = resolve(__dirname, '../../examples/fonts');

const fontsAvailable =
  existsSync(resolve(fontsDir, 'EBGaramond-Regular.ttf')) &&
  existsSync(resolve(fontsDir, 'EBGaramond-Italic.ttf')) &&
  existsSync(resolve(fontsDir, 'EBGaramond-Bold.ttf'));

async function createMeasurer(): Promise<TextMeasurer> {
  const fm = new FontManager();
  await fm.loadFont('body', readFileSync(resolve(fontsDir, 'EBGaramond-Regular.ttf')).buffer);
  await fm.loadFont('bodyItalic', readFileSync(resolve(fontsDir, 'EBGaramond-Italic.ttf')).buffer);
  await fm.loadFont('bodyBold', readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')).buffer);
  await fm.loadFont('heading', readFileSync(resolve(fontsDir, 'EBGaramond-Bold.ttf')).buffer);
  return new TextMeasurer(fm, 11);
}

describe.skipIf(!fontsAvailable)('TextMeasurer', () => {
  describe('ligature handling', () => {
    it('measures "fi" ligature consistently (fi vs f+i)', async () => {
      const m = await createMeasurer();
      const fiWidth = m.measureText('fi', 'body');
      const fWidth = m.measureText('f', 'body');
      const iWidth = m.measureText('i', 'body');
      // "fi" should NOT equal f+i if ligature is applied — the ligature glyph is narrower
      // This test documents that we account for ligatures
      expect(fiWidth).not.toBeCloseTo(fWidth + iWidth, 1);
      expect(fiWidth).toBeLessThan(fWidth + iWidth);
    });

    it('measures "fl" ligature consistently', async () => {
      const m = await createMeasurer();
      const flWidth = m.measureText('fl', 'body');
      const fWidth = m.measureText('f', 'body');
      const lWidth = m.measureText('l', 'body');
      // fl ligature should be narrower than f+l separate
      expect(flWidth).toBeLessThan(fWidth + lWidth);
    });

    it('measures words with "fi" using ligature width', async () => {
      const m = await createMeasurer();
      // Words containing common ligatures
      const words = ['fixed', 'justified', 'justification', 'first', 'find', 'office', 'float'];
      for (const word of words) {
        const width = m.measureText(word, 'body');
        // Width should be positive and reasonable
        expect(width).toBeGreaterThan(0);
        // Build expected width from individual chars (no ligatures)
        let charByCharWidth = 0;
        for (let i = 0; i < word.length; i++) {
          charByCharWidth += m.measureText(word[i], 'body');
        }
        // Ligature-aware measurement should be narrower than char-by-char
        // for words containing fi/fl
        if (word.includes('fi') || word.includes('fl')) {
          expect(width).toBeLessThan(charByCharWidth);
        }
      }
    });

    it('measureRuns applies ligatures for styled text', async () => {
      const m = await createMeasurer();
      const runs = [{ text: 'justified text', bold: false, italic: false }];
      const measured = m.measureRuns(runs);

      // "justified" token should use ligature-aware width
      const justifiedToken = measured.find(w => w.text === 'justified');
      expect(justifiedToken).toBeDefined();

      // Measure char-by-char for comparison
      let charByChar = 0;
      for (const ch of 'justified') {
        charByChar += m.measureText(ch, 'body');
      }
      // The token width should be less than char-by-char sum (ligature is narrower)
      expect(justifiedToken!.width).toBeLessThan(charByChar);
    });

    it('measureRuns handles ligatures in italic text', async () => {
      const m = await createMeasurer();
      const runs = [{ text: 'finding', bold: false, italic: true }];
      const measured = m.measureRuns(runs);

      const token = measured.find(w => w.text === 'finding');
      expect(token).toBeDefined();
      expect(token!.fontStyle).toBe('bodyItalic');

      // Should use ligature-aware width
      let charByChar = 0;
      for (const ch of 'finding') {
        charByChar += m.measureText(ch, 'bodyItalic');
      }
      expect(token!.width).toBeLessThan(charByChar);
    });
  });

  describe('basic measurement', () => {
    it('measures a space', async () => {
      const m = await createMeasurer();
      const spaceWidth = m.measureSpace('body');
      expect(spaceWidth).toBeGreaterThan(0);
      expect(spaceWidth).toBeLessThan(5); // reasonable at 11pt
    });

    it('measureRuns splits on whitespace', async () => {
      const m = await createMeasurer();
      const runs = [{ text: 'hello world', bold: false, italic: false }];
      const measured = m.measureRuns(runs);
      expect(measured).toHaveLength(3); // "hello", " ", "world"
      expect(measured[0].text).toBe('hello');
      expect(measured[1].text).toBe(' ');
      expect(measured[2].text).toBe('world');
    });

    it('measureRuns assigns correct fontStyle', async () => {
      const m = await createMeasurer();
      const runs = [
        { text: 'normal ', bold: false, italic: false },
        { text: 'italic ', bold: false, italic: true },
        { text: 'bold', bold: true, italic: false },
      ];
      const measured = m.measureRuns(runs);
      const content = measured.filter(w => w.text.trim() !== '');
      expect(content[0].fontStyle).toBe('body');
      expect(content[1].fontStyle).toBe('bodyItalic');
      expect(content[2].fontStyle).toBe('bodyBold');
    });
  });
});
