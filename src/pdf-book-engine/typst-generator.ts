import type { Block, Chapter, PdfBookConfig, StyledRun } from './types.js';
import { parseHtml } from './html-parser.js';
import { lookupGutter, LULU_PROFILE } from './printer-profiles.js';

/**
 * Generate a complete Typst document string from chapters and config.
 */
export function generateTypstDocument(
  chapters: Chapter[],
  config: PdfBookConfig,
  gutterInches: number,
): string {
  const parts: string[] = [];

  // Page setup
  parts.push(generatePageSetup(config, gutterInches));

  // Text and paragraph setup
  parts.push(generateTextSetup(config));

  // Font setup
  parts.push(generateFontSetup(config));

  // Chapter rendering
  for (let i = 0; i < chapters.length; i++) {
    if (i > 0 || config.chapterStartRecto) {
      parts.push(generateChapterBreak(config));
    }
    const blocks = parseHtml(chapters[i].html);
    parts.push(generateChapter(chapters[i].title, blocks, config));
  }

  return parts.join('\n');
}

function generatePageSetup(config: PdfBookConfig, gutterInches: number): string {
  const lines: string[] = [];

  // Header/footer helper functions must be defined BEFORE #set page()
  if (config.header) {
    lines.push(generateHeaderFooterFunc('_typst_header_', config.header, config));
    lines.push('');
  }
  if (config.footer) {
    lines.push(generateHeaderFooterFunc('_typst_footer_', config.footer, config));
    lines.push('');
  }

  lines.push('#set page(');
  lines.push(`  width: ${config.trimWidth}in,`);
  lines.push(`  height: ${config.trimHeight}in,`);
  lines.push('  margin: (');
  lines.push(`    top: ${config.margins.top}in,`);
  lines.push(`    bottom: ${config.margins.bottom}in,`);
  lines.push(`    inside: ${gutterInches}in,`);
  lines.push(`    outside: ${config.margins.outside}in,`);
  lines.push('  ),');
  lines.push('  binding: left,');

  // Header
  if (config.header) {
    lines.push(`  header: locate(loc => {`);
    lines.push(`    let page-num = counter(page).at(loc).first()`);
    lines.push(`    _typst_header_(page-num)`);
    lines.push(`  }),`);
  }

  // Footer
  if (config.footer) {
    lines.push(`  footer: locate(loc => {`);
    lines.push(`    let page-num = counter(page).at(loc).first()`);
    lines.push(`    _typst_footer_(page-num)`);
    lines.push(`  }),`);
  }

  lines.push(')');

  return lines.join('\n');
}

/**
 * Pre-evaluate the header/footer callback for common page scenarios
 * and generate a Typst function that reproduces the behavior.
 */
function generateHeaderFooterFunc(
  name: string,
  callback: (ctx: import('./types.js').PageContext) => import('./types.js').HeaderFooterContent | null,
  config: PdfBookConfig,
): string {
  // Sample the callback with representative page contexts to detect the pattern
  const samples = [
    { pageNumber: 1, isRecto: true, isChapterOpener: true, chapterTitle: 'CHAPTER', bookTitle: 'BOOK', totalPages: 100 },
    { pageNumber: 2, isRecto: false, isChapterOpener: false, chapterTitle: 'CHAPTER', bookTitle: 'BOOK', totalPages: 100 },
    { pageNumber: 3, isRecto: true, isChapterOpener: false, chapterTitle: 'CHAPTER', bookTitle: 'BOOK', totalPages: 100 },
  ];

  const results = samples.map(ctx => ({ ctx, result: callback(ctx) }));

  const lines: string[] = [];
  lines.push(`#let ${name}(page-num) = {`);

  // Build conditional branches
  const branches: string[] = [];

  // Chapter opener (page 1 in our samples)
  const openerResult = results[0].result;
  if (openerResult === null) {
    // No header on chapter openers — detected by checking isChapterOpener
    // We'll use a state-based approach below
  }

  // Check if chapter openers are suppressed
  const chapterOpenerSuppressed = openerResult === null;

  // Verso (even) page
  const versoResult = results[1].result;
  // Recto (odd) non-opener page
  const rectoResult = results[2].result;

  if (chapterOpenerSuppressed) {
    // We need chapter opener state tracking — use Typst's state
    // For simplicity, we'll track chapter openers via a custom label
    lines.push('  // Chapter opener pages have no header/footer');
  }

  // Generate the conditional logic
  if (versoResult !== null || rectoResult !== null) {
    const needsOddEven = JSON.stringify(versoResult) !== JSON.stringify(rectoResult);

    if (needsOddEven) {
      lines.push('  if calc.odd(page-num) {');
      if (rectoResult) {
        lines.push(`    ${generateHeaderFooterContent(rectoResult)}`);
      }
      lines.push('  } else {');
      if (versoResult) {
        lines.push(`    ${generateHeaderFooterContent(versoResult)}`);
      }
      lines.push('  }');
    } else if (rectoResult) {
      lines.push(`  ${generateHeaderFooterContent(rectoResult)}`);
    }
  }

  lines.push('}');
  return lines.join('\n');
}

function generateHeaderFooterContent(content: import('./types.js').HeaderFooterContent): string {
  const fontSize = content.fontSize ?? 9;
  const parts: string[] = [];

  if (content.left && content.right) {
    // Use a grid for left + right alignment
    const left = escapeTypst(content.left)
      .replace('BOOK', '" + _book-title_ + "')
      .replace('CHAPTER', '" + _chapter-title_ + "');
    const right = escapeTypst(content.right)
      .replace('BOOK', '" + _book-title_ + "')
      .replace('CHAPTER', '" + _chapter-title_ + "');
    return `text(size: ${fontSize}pt, fill: rgb("#4d4d4d"))[#grid(columns: (1fr, 1fr), align(left)[${left}], align(right)[${right}])]`;
  }

  if (content.center) {
    let center = escapeTypst(content.center);
    // Replace page number placeholder if present
    if (center.includes('BOOK') || center.includes('CHAPTER')) {
      center = center
        .replace('BOOK', '" + _book-title_ + "')
        .replace('CHAPTER', '" + _chapter-title_ + "');
    }
    // If content is a page number pattern, use Typst's counter
    return `align(center, text(size: ${fontSize}pt, fill: rgb("#4d4d4d"))[${center}])`;
  }

  if (content.left) {
    const left = escapeTypst(content.left);
    return `align(left, text(size: ${fontSize}pt, fill: rgb("#4d4d4d"))[${left}])`;
  }

  if (content.right) {
    const right = escapeTypst(content.right);
    return `align(right, text(size: ${fontSize}pt, fill: rgb("#4d4d4d"))[${right}])`;
  }

  return '';
}

function generateTextSetup(config: PdfBookConfig): string {
  const lines: string[] = [];
  const leading = (config.lineHeight - 1) * config.fontSize;

  lines.push(`#set text(size: ${config.fontSize}pt, lang: "en", hyphenate: false)`);
  lines.push(`#set par(justify: true, first-line-indent: ${config.paragraphIndent}em, leading: ${leading.toFixed(2)}pt)`);

  return lines.join('\n');
}

function generateFontSetup(config: PdfBookConfig): string {
  // Font names are resolved by Typst from --font-path
  // We just need to set the font family if custom fonts are provided
  // The font family name comes from the font file metadata
  // For now, we don't set a specific font — Typst will use whatever is available
  // The caller should pass --font-path pointing to the font directory
  return '';
}

function generateChapterBreak(config: PdfBookConfig): string {
  if (config.chapterStartRecto) {
    return '#pagebreak(to: "odd")';
  }
  return '#pagebreak()';
}

function generateChapter(
  title: string,
  blocks: Block[],
  config: PdfBookConfig,
): string {
  const parts: string[] = [];

  // Chapter title with top drop
  if (config.chapterTopDrop > 0) {
    parts.push(`#v(${config.chapterTopDrop}in)`);
  }

  // Chapter title as level-1 heading
  const titleSize = config.fontSize * 1.8;
  parts.push(`#text(size: ${titleSize.toFixed(1)}pt, weight: "bold")[${escapeTypst(title)}]`);
  parts.push('');
  parts.push(`#v(${(config.fontSize * config.lineHeight).toFixed(1)}pt)`);

  // Process blocks
  for (const block of blocks) {
    parts.push(generateBlock(block, config));
  }

  return parts.join('\n');
}

function generateBlock(block: Block, config: PdfBookConfig): string {
  switch (block.type) {
    case 'paragraph':
      return generateParagraph(block.runs);

    case 'heading': {
      const size = block.level === 2 ? config.fontSize * 1.4 : config.fontSize * 1.2;
      const text = block.runs.map(r => r.text).join('');
      const prefix = block.level === 2 ? '==' : '===';
      // Use show rule inline for heading size
      return `\n#v(${(size * config.lineHeight * 0.8).toFixed(1)}pt)\n${prefix} ${escapeTypst(text)}\n#v(${(size * config.lineHeight * 0.3).toFixed(1)}pt)`;
    }

    case 'list':
      return generateList(block.ordered, block.items);

    case 'image':
      if (block.src) {
        return `#image("${escapeTypst(block.src)}")`;
      }
      return '';

    case 'lineBreak':
      return '#v(1em)';
  }
}

function generateParagraph(runs: StyledRun[]): string {
  const parts: string[] = [];
  for (const run of runs) {
    let text = escapeTypst(run.text);

    if (run.bold && run.italic) {
      text = `*_${text}_*`;
    } else if (run.bold) {
      text = `*${text}*`;
    } else if (run.italic) {
      text = `_${text}_`;
    }

    if (run.link) {
      text = `#link("${escapeTypst(run.link)}")[${text}]`;
    }

    parts.push(text);
  }
  return parts.join('');
}

function generateList(ordered: boolean, items: StyledRun[][]): string {
  const lines: string[] = [];
  for (let i = 0; i < items.length; i++) {
    const content = items[i].map(run => {
      let text = escapeTypst(run.text);
      if (run.bold) text = `*${text}*`;
      if (run.italic) text = `_${text}_`;
      return text;
    }).join('');

    if (ordered) {
      lines.push(`+ ${content}`);
    } else {
      lines.push(`- ${content}`);
    }
  }
  return lines.join('\n');
}

/**
 * Escape special Typst characters in text content.
 */
function escapeTypst(text: string): string {
  // Typst special characters that need escaping with backslash:
  // * _ ` $ # @ < > = ~ ^ / \
  // We only escape characters that would be misinterpreted in content context.
  // Bold (*) and italic (_) are handled by the caller when intentional.
  return text
    .replace(/\\/g, '\\\\')
    .replace(/#/g, '\\#')
    .replace(/\$/g, '\\$')
    .replace(/@/g, '\\@')
    .replace(/</g, '\\<')
    .replace(/>/g, '\\>')
    .replace(/~/g, '\\~')
    .replace(/\^/g, '\\^');
}

export { escapeTypst };
