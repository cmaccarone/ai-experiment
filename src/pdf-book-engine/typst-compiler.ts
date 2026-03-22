import { writeFileSync, readFileSync, mkdtempSync, rmSync, mkdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { compile as typstCompile } from 'typst';

export interface CompileOptions {
  /** Directories containing font files (.ttf, .otf) */
  fontPaths?: string[];
  /** Font buffers to write to a temp dir and pass as --font-path */
  fontBuffers?: Map<string, Uint8Array>;
}

/**
 * Compile a Typst source string to PDF bytes.
 *
 * Writes the source to a temp file, invokes the Typst CLI compiler,
 * reads the resulting PDF, and cleans up.
 */
export async function compileTypst(
  source: string,
  options: CompileOptions = {},
): Promise<Uint8Array> {
  const dir = mkdtempSync(join(tmpdir(), 'typst-book-'));
  const inputPath = join(dir, 'book.typ');
  const outputPath = join(dir, 'book.pdf');

  try {
    writeFileSync(inputPath, source);

    // If font buffers are provided, write them to a temp directory
    let fontDir: string | undefined;
    if (options.fontBuffers && options.fontBuffers.size > 0) {
      fontDir = join(dir, 'fonts');
      mkdirSync(fontDir);
      for (const [name, buffer] of options.fontBuffers) {
        writeFileSync(join(fontDir, name), buffer);
      }
    }

    // Build compile options — typst npm package only supports a single fontPath
    const compileOpts: Record<string, any> = {};
    const fontPath = fontDir ?? options.fontPaths?.[0];
    if (fontPath) {
      compileOpts.fontPath = fontPath;
    }

    await typstCompile(inputPath, outputPath, compileOpts);
    return new Uint8Array(readFileSync(outputPath));
  } finally {
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}
