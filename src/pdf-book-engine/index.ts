import type { Chapter, PdfBookConfig, FontStyle } from './types.js';
import { FontManager } from './font-manager.js';
import { PdfWriter } from './pdf-writer.js';
import { PdfRenderer } from './pdf-renderer.js';
import { layoutBook } from './layout-engine.js';
import { validateConfig, validatePageCount } from './validator.js';
import { LULU_PROFILE } from './printer-profiles.js';

// Re-export public types
export type {
  Chapter,
  PdfBookConfig,
  PageContext,
  HeaderFooterContent,
  PrinterProfile,
  ValidationWarning,
} from './types.js';

export { LULU_PROFILE, KDP_PROFILE, INGRAM_PROFILE } from './printer-profiles.js';

/**
 * Generate a print-ready PDF book from structured chapter data.
 */
export async function generateBook(
  chapters: Chapter[],
  config: PdfBookConfig,
  images: Record<string, ArrayBuffer> = {},
): Promise<Uint8Array> {
  const printer = config.printer ?? LULU_PROFILE;

  // Validate config
  const warnings = validateConfig(config);
  if (warnings.length > 0) {
    console.warn('[pdf-book-engine] Validation warnings:');
    for (const w of warnings) {
      console.warn(`  [${w.code}] ${w.message}`);
    }
  }

  // Load fonts
  const fontManager = new FontManager();
  const fontEntries: Array<[FontStyle, string | ArrayBuffer | undefined]> = [
    ['body', config.fonts.body],
    ['bodyItalic', config.fonts.bodyItalic],
    ['bodyBold', config.fonts.bodyBold],
    ['heading', config.fonts.heading],
  ];

  for (const [style, source] of fontEntries) {
    if (source) {
      await fontManager.loadFont(style, source);
    }
  }

  // Run layout pipeline (includes convergence loop)
  const { pages, warnings: layoutWarnings } = await layoutBook(
    chapters,
    config,
    images,
    fontManager,
  );

  if (layoutWarnings.length > 0) {
    for (const w of layoutWarnings) {
      console.warn(`[pdf-book-engine] ${w}`);
    }
  }

  // Validate page count
  const pageCountWarnings = validatePageCount(pages.length, printer);
  for (const w of pageCountWarnings) {
    console.warn(`[pdf-book-engine] [${w.code}] ${w.message}`);
  }

  // Create PDF
  const writer = await PdfWriter.create();
  await writer.embedFonts(fontManager);

  // Embed images
  const embeddedImages = new Map<string, any>();
  const doc = writer.getDocument();
  for (const [src, buffer] of Object.entries(images)) {
    try {
      // Try PNG first, then JPG
      let img;
      try {
        img = await doc.embedPng(buffer);
      } catch {
        img = await doc.embedJpg(buffer);
      }
      embeddedImages.set(src, img);
    } catch {
      console.warn(`[pdf-book-engine] Failed to embed image: ${src}`);
    }
  }

  // Determine book title (use first chapter title as fallback)
  const bookTitle = chapters[0]?.title ?? 'Untitled';

  // Render all pages
  const renderer = new PdfRenderer(writer, config, embeddedImages);
  for (const page of pages) {
    renderer.renderPage(page, bookTitle, pages.length);
  }

  return writer.save();
}
