import { PDFDocument, PDFFont, PDFPage, rgb, StandardFonts } from 'pdf-lib';
import type { FontManager } from './font-manager.js';
import type { FontStyle } from './types.js';

/**
 * Manages the pdf-lib document, font embedding, and page creation.
 */
export class PdfWriter {
  private doc: PDFDocument;
  private embeddedFonts = new Map<FontStyle, PDFFont>();

  private constructor(doc: PDFDocument) {
    this.doc = doc;
  }

  static async create(): Promise<PdfWriter> {
    const doc = await PDFDocument.create();
    return new PdfWriter(doc);
  }

  async embedFonts(fontManager: FontManager): Promise<void> {
    const styles: FontStyle[] = ['body', 'bodyItalic', 'bodyBold', 'heading'];

    for (const style of styles) {
      if (fontManager.hasFont(style)) {
        try {
          const font = fontManager.getFont(style);
          const fontBuffer = font.toArrayBuffer();
          const embedded = await this.doc.embedFont(fontBuffer);
          this.embeddedFonts.set(style, embedded);
        } catch {
          // If custom font embedding fails, fall back to standard font
          await this.embedFallbackFont(style);
        }
      } else {
        await this.embedFallbackFont(style);
      }
    }
  }

  private async embedFallbackFont(style: FontStyle): Promise<void> {
    let stdFont: StandardFonts;
    switch (style) {
      case 'bodyBold':
        stdFont = StandardFonts.TimesRomanBold;
        break;
      case 'bodyItalic':
        stdFont = StandardFonts.TimesRomanItalic;
        break;
      case 'heading':
        stdFont = StandardFonts.HelveticaBold;
        break;
      default:
        stdFont = StandardFonts.TimesRoman;
    }
    const embedded = await this.doc.embedFont(stdFont);
    this.embeddedFonts.set(style, embedded);
  }

  getFont(style: FontStyle): PDFFont {
    const font = this.embeddedFonts.get(style);
    if (!font) {
      // Fallback to body
      return this.embeddedFonts.get('body')!;
    }
    return font;
  }

  addPage(width: number, height: number): PDFPage {
    return this.doc.addPage([width, height]);
  }

  getDocument(): PDFDocument {
    return this.doc;
  }

  async save(): Promise<Uint8Array> {
    return this.doc.save();
  }
}
