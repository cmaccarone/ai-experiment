import * as opentypeModule from 'opentype.js';

// Handle both ESM default and CJS exports
const opentype = (opentypeModule as any).default ?? opentypeModule;
import type { FontStyle } from './types.js';

export interface FontMetrics {
  unitsPerEm: number;
  ascender: number;
  descender: number;
  font: opentype.Font;
  // Cached glyph widths: codepoint → advance width in font units
  widthCache: Map<number, number>;
  // Cached kerning: "cp1,cp2" → kerning value in font units
  kernCache: Map<string, number>;
}

export class FontManager {
  private fonts = new Map<FontStyle, FontMetrics>();

  async loadFont(style: FontStyle, source: string | ArrayBuffer): Promise<void> {
    let font: opentype.Font;

    if (typeof source === 'string') {
      font = await opentype.load(source);
    } else {
      // Ensure we have a true ArrayBuffer (Node Buffer won't work with DataView)
      const buffer = source instanceof ArrayBuffer
        ? source
        : new Uint8Array(source as any).buffer;
      font = opentype.parse(buffer);
    }

    const metrics: FontMetrics = {
      unitsPerEm: font.unitsPerEm,
      ascender: font.ascender,
      descender: font.descender,
      font,
      widthCache: new Map(),
      kernCache: new Map(),
    };

    this.fonts.set(style, metrics);
  }

  getMetrics(style: FontStyle): FontMetrics {
    const metrics = this.fonts.get(style);
    if (!metrics) {
      // Fallback chain: bodyBold/bodyItalic → body
      if (style === 'bodyBold' || style === 'bodyItalic') {
        const body = this.fonts.get('body');
        if (body) return body;
      }
      // heading → body
      if (style === 'heading') {
        const body = this.fonts.get('body');
        if (body) return body;
      }
      throw new Error(`Font not loaded for style: ${style}`);
    }
    return metrics;
  }

  /**
   * Get the advance width of a codepoint in font units.
   * Results are cached for performance.
   */
  getGlyphWidth(style: FontStyle, codepoint: number): number {
    const metrics = this.getMetrics(style);
    const cached = metrics.widthCache.get(codepoint);
    if (cached !== undefined) return cached;

    const glyph = metrics.font.charToGlyph(String.fromCodePoint(codepoint));
    const width = glyph.advanceWidth ?? 0;
    metrics.widthCache.set(codepoint, width);
    return width;
  }

  /**
   * Get kerning between two codepoints in font units.
   */
  getKerning(style: FontStyle, cp1: number, cp2: number): number {
    const metrics = this.getMetrics(style);
    const key = `${cp1},${cp2}`;
    const cached = metrics.kernCache.get(key);
    if (cached !== undefined) return cached;

    const glyph1 = metrics.font.charToGlyph(String.fromCodePoint(cp1));
    const glyph2 = metrics.font.charToGlyph(String.fromCodePoint(cp2));
    const kern = metrics.font.getKerningValue(glyph1, glyph2);
    metrics.kernCache.set(key, kern);
    return kern;
  }

  /**
   * Get the line height metrics for a given font style at a given size.
   */
  getLineMetrics(style: FontStyle, fontSize: number): { ascent: number; descent: number } {
    const metrics = this.getMetrics(style);
    const scale = fontSize / metrics.unitsPerEm;
    return {
      ascent: metrics.ascender * scale,
      descent: Math.abs(metrics.descender) * scale,
    };
  }

  hasFont(style: FontStyle): boolean {
    return this.fonts.has(style);
  }

  getFont(style: FontStyle): opentype.Font {
    return this.getMetrics(style).font;
  }
}
