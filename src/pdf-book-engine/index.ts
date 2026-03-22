import type { Chapter, PdfBookConfig, FontStyle } from './types.js';
import { validateConfig, validatePageCount } from './validator.js';
import { lookupGutter, LULU_PROFILE } from './printer-profiles.js';
import { generateTypstDocument } from './typst-generator.js';
import { compileTypst, type CompileOptions } from './typst-compiler.js';
import { writeFileSync, mkdtempSync, rmSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { tmpdir } from 'os';

// Re-export public types
export type {
  Chapter,
  PdfBookConfig,
  HeaderFooterConfig,
  PrinterProfile,
  ValidationWarning,
} from './types.js';

export { LULU_PROFILE, KDP_PROFILE, INGRAM_PROFILE } from './printer-profiles.js';

/**
 * Generate a print-ready PDF book from structured chapter data.
 *
 * Uses Typst for typesetting — provides Knuth-Plass line breaking,
 * widow/orphan control, and high-quality PDF output.
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

  // Resolve font paths for Typst
  const { fontDir, cleanup } = await prepareFonts(config);

  try {
    // Two-pass gutter convergence:
    // 1. First pass with estimated gutter to get page count
    // 2. Second pass with correct gutter (if it changed)
    const estimatedPages = chapters.length * 20;
    let gutter = lookupGutter(printer, estimatedPages);

    // First pass
    let typstSource = generateTypstDocument(chapters, config, gutter);
    const compileOpts: CompileOptions = {};
    if (fontDir) {
      compileOpts.fontPaths = [fontDir];
    }

    let pdfBytes = await compileTypst(typstSource, compileOpts);

    // Estimate page count from PDF (count page objects)
    const pageCount = countPdfPages(pdfBytes);
    const newGutter = lookupGutter(printer, pageCount);

    // Second pass if gutter changed
    if (newGutter !== gutter) {
      gutter = newGutter;
      typstSource = generateTypstDocument(chapters, config, gutter);
      pdfBytes = await compileTypst(typstSource, compileOpts);
    }

    // Validate page count
    const finalPageCount = countPdfPages(pdfBytes);
    const pageCountWarnings = validatePageCount(finalPageCount, printer);
    for (const w of pageCountWarnings) {
      console.warn(`[pdf-book-engine] [${w.code}] ${w.message}`);
    }

    return pdfBytes;
  } finally {
    cleanup();
  }
}

/**
 * Prepare fonts for Typst compilation.
 * If fonts are provided as ArrayBuffers, write them to a temp directory.
 * If fonts are provided as file paths, use the directory of the first font.
 */
async function prepareFonts(config: PdfBookConfig): Promise<{ fontDir: string | null; cleanup: () => void }> {
  const fontEntries: Array<{ name: string; source: string | ArrayBuffer | undefined }> = [
    { name: 'body.ttf', source: config.fonts.body },
    { name: 'body-italic.ttf', source: config.fonts.bodyItalic },
    { name: 'body-bold.ttf', source: config.fonts.bodyBold },
    { name: 'heading.ttf', source: config.fonts.heading },
  ];

  const hasBuffers = fontEntries.some(e => e.source instanceof ArrayBuffer);
  const hasPaths = fontEntries.some(e => typeof e.source === 'string');

  if (hasBuffers) {
    // Write buffers to temp dir
    const dir = mkdtempSync(join(tmpdir(), 'typst-fonts-'));
    for (const entry of fontEntries) {
      if (entry.source instanceof ArrayBuffer) {
        writeFileSync(join(dir, entry.name), new Uint8Array(entry.source));
      }
    }
    return {
      fontDir: dir,
      cleanup: () => {
        try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
      },
    };
  }

  if (hasPaths) {
    // Use the directory of the first font path
    const firstPath = fontEntries.find(e => typeof e.source === 'string')?.source as string;
    return { fontDir: dirname(firstPath), cleanup: () => {} };
  }

  return { fontDir: null, cleanup: () => {} };
}

/**
 * Count pages in a PDF by counting /Type /Page occurrences.
 */
function countPdfPages(pdfBytes: Uint8Array): number {
  // Search for "/Type /Page" (not "/Type /Pages") in the PDF
  const text = new TextDecoder('latin1').decode(pdfBytes);
  const matches = text.match(/\/Type\s*\/Page[^s]/g);
  return matches ? matches.length : 1;
}
