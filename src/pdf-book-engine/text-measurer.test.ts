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
  describe('ligature handling (deliberately disabled)', () => {
    // pdf-lib/fontkit has a bug where GSUB ligature substitution creates
    // visual gaps inside words when rendered via drawText. To keep
    // measurement and rendering consistent, we deliberately measure
    // char-by-char WITHOUT ligatures. The renderer also draws char-by-char
    // to avoid triggering fontkit's ligature bug.

    it('measures "fi" as f+i (no ligature substitution)', async () => {
      const m = await createMeasurer();
      const fiWidth = m.measureText('fi', 'body');
      const fWidth = m.measureText('f', 'body');
      const iWidth = m.measureText('i', 'body');
      // Measurement should equal f+i since we skip ligatures
      expect(fiWidth).toBeCloseTo(fWidth + iWidth, 5);
    });

    it('measures "fl" as f+l (no ligature substitution)', async () => {
      const m = await createMeasurer();
      const flWidth = m.measureText('fl', 'body');
      const fWidth = m.measureText('f', 'body');
      const lWidth = m.measureText('l', 'body');
      expect(flWidth).toBeCloseTo(fWidth + lWidth, 5);
    });

    it('word measurement equals sum of character measurements', async () => {
      const m = await createMeasurer();
      // Words containing common ligature pairs (fi, fl)
      const words = ['fixed', 'justified', 'justification', 'first', 'float', 'office'];
      for (const word of words) {
        const wordWidth = m.measureText(word, 'body');
        let charSum = 0;
        for (let i = 0; i < word.length; i++) {
          charSum += m.measureText(word[i], 'body');
        }
        // Word width should match char-by-char sum (no ligature applied)
        // Small tolerance for kerning differences
        expect(wordWidth).toBeCloseTo(charSum, 1);
      }
    });

    it('measureRuns width matches char-by-char for ligature words', async () => {
      const m = await createMeasurer();
      const runs = [{ text: 'justified text', bold: false, italic: false }];
      const measured = m.measureRuns(runs);

      const justifiedToken = measured.find(w => w.text === 'justified');
      expect(justifiedToken).toBeDefined();

      let charByChar = 0;
      for (const ch of 'justified') {
        charByChar += m.measureText(ch, 'body');
      }
      // Should match (no ligature savings)
      expect(justifiedToken!.width).toBeCloseTo(charByChar, 1);
    });

    it('measureRuns italic width matches char-by-char', async () => {
      const m = await createMeasurer();
      const runs = [{ text: 'finding', bold: false, italic: true }];
      const measured = m.measureRuns(runs);

      const token = measured.find(w => w.text === 'finding');
      expect(token).toBeDefined();
      expect(token!.fontStyle).toBe('bodyItalic');

      let charByChar = 0;
      for (const ch of 'finding') {
        charByChar += m.measureText(ch, 'bodyItalic');
      }
      expect(token!.width).toBeCloseTo(charByChar, 1);
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
