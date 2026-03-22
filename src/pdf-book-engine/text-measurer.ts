import type { FontStyle, MeasuredWord, StyledRun } from './types.js';
import type { FontManager } from './font-manager.js';

/**
 * Measures text runs using cached font metrics.
 * Splits runs into words and computes widths for the line breaker.
 */
export class TextMeasurer {
  constructor(
    private fontManager: FontManager,
    private fontSize: number,
  ) {}

  /**
   * Measure a string of text in a given font style.
   * Returns width in points.
   */
  measureText(text: string, style: FontStyle): number {
    const metrics = this.fontManager.getMetrics(style);
    const scale = this.fontSize / metrics.unitsPerEm;
    let width = 0;

    // Measure character-by-character to match pdf-lib's actual rendering.
    // pdf-lib/fontkit has a bug where GSUB ligature substitution (fi, fl)
    // creates visual gaps inside words when rendered via drawText. So we
    // deliberately skip ligatures here to keep measurement and rendering
    // consistent. See text-measurer.test.ts for details.
    for (let i = 0; i < text.length; i++) {
      const cp = text.codePointAt(i)!;
      const glyphWidth = this.fontManager.getGlyphWidth(style, cp);
      width += glyphWidth;

      // Add kerning
      if (i + 1 < text.length) {
        const nextCp = text.codePointAt(i + 1)!;
        width += this.fontManager.getKerning(style, cp, nextCp);
      }

      // Handle surrogate pairs
      if (cp > 0xffff) i++;
    }

    return width * scale;
  }

  /**
   * Measure the width of a single space character in the given font style.
   */
  measureSpace(style: FontStyle): number {
    return this.measureText(' ', style);
  }

  /**
   * Convert an array of StyledRuns into MeasuredWords for the line breaker.
   * Words are split on whitespace boundaries. Whitespace is tracked separately.
   */
  measureRuns(runs: StyledRun[]): MeasuredWord[] {
    const words: MeasuredWord[] = [];

    for (const run of runs) {
      const fontStyle = run.bold ? 'bodyBold' as FontStyle : run.italic ? 'bodyItalic' as FontStyle : 'body' as FontStyle;

      // Split into words preserving whitespace boundaries
      const parts = run.text.split(/(\s+)/);

      for (const part of parts) {
        if (part === '') continue;

        const width = this.measureText(part, fontStyle);
        words.push({
          text: part,
          width,
          fontStyle,
          ...(run.link ? { link: run.link } : {}),
        });
      }
    }

    return words;
  }
}
