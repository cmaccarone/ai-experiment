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

  // Table of contents — use a heading that doesn't appear in the outline itself,
  // and suppress headers/footers on TOC pages by wrapping in a set page scope.
  if (config.tableOfContents !== false) {
    parts.push('#outline(depth: 1)');
    parts.push('#pagebreak()');
  }

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

  if (config.header) {
    lines.push(`  header: locate(loc => {`);
    lines.push(generateHeaderFooterBody(config.header));
    lines.push(`  }),`);
  }

  if (config.footer) {
    lines.push(`  footer: locate(loc => {`);
    lines.push(generateHeaderFooterBody(config.footer));
    lines.push(`  }),`);
  }

  lines.push(')');

  return lines.join('\n');
}

/**
 * Build a Typst content expression from a string with PAGE / CHAPTER placeholders.
 * Uses variables `n` (page number) and `chapter-title` that are set in the
 * enclosing locate block.
 */
function buildContentExpr(text: string): string {
  return text.split(/(PAGE|CHAPTER)/).map(part => {
    if (part === 'PAGE') return '#str(n)';
    if (part === 'CHAPTER') return '#chapter-title';
    return escapeTypst(part);
  }).join('');
}

/**
 * Generate the body of a `locate(loc => { ... })` block for a header or footer.
 * Compatible with Typst 0.10 — uses locate + counter.at(loc) + query(..., loc).
 */
function generateHeaderFooterBody(cfg: import('./types.js').HeaderFooterConfig): string {
  const lines: string[] = [];
  const fontSize = cfg.fontSize ?? 9;

  lines.push(`    let n = counter(page).at(loc).first()`);

  // Query only real chapter headings (labeled <book-chapter>), not the outline title
  const needsChapter = [cfg.outside, cfg.inside, cfg.center]
    .some(s => s && s.includes('CHAPTER'));

  if (cfg.hideOnChapterOpener !== false) {
    // Skip on chapter opener pages and any pages before the first chapter (e.g. TOC)
    lines.push(`    let book-chapters = query(<book-chapter>, loc)`);
    lines.push(`    let chapter-pages = book-chapters.map(h => h.location().page())`);
    lines.push(`    if n in chapter-pages { return }`);
    lines.push(`    if chapter-pages.len() > 0 and n < chapter-pages.first() { return }`);
  }

  if (needsChapter) {
    if (cfg.hideOnChapterOpener !== false) {
      // book-chapters already queried above
      lines.push(`    let prev-chapters = book-chapters.filter(h => h.location().page() <= n)`);
    } else {
      lines.push(`    let prev-chapters = query(<book-chapter>, loc).filter(h => h.location().page() <= n)`);
    }
    lines.push(`    let chapter-title = if prev-chapters.len() > 0 { prev-chapters.last().body } else { [] }`);
  }

  if (cfg.center) {
    const content = buildContentExpr(cfg.center);
    lines.push(`    align(center, text(size: ${fontSize}pt)[${content}])`);
  } else if (cfg.outside || cfg.inside) {
    const outside = cfg.outside ? buildContentExpr(cfg.outside) : '';
    const inside = cfg.inside ? buildContentExpr(cfg.inside) : '';
    const sepText = cfg.separator ? escapeTypst(cfg.separator) : '';

    // Recto (odd): inside on left, outside on right
    // Verso (even): outside on left, inside on right
    const rectoLeft = inside;
    const rectoRight = outside;
    const versoLeft = outside;
    const versoRight = inside;

    const buildGroup = (parts: string[]): string => {
      const filtered = parts.filter(Boolean);
      if (filtered.length === 1) return filtered[0];
      return `grid(columns: (${filtered.map(() => 'auto').join(', ')}), column-gutter: 6pt, ${filtered.join(', ')})`;
    };

    const txt = (content: string) => `text(size: ${fontSize}pt)[${content}]`;

    const buildRow = (left: string, right: string, side: 'left' | 'right'): string => {
      if (left && right) {
        const parts = sepText ? [txt(left), txt(sepText), txt(right)] : [txt(left), txt(right)];
        return `align(${side}, ${buildGroup(parts)})`;
      }
      if (left) return `align(left, ${txt(left)})`;
      if (right) return `align(right, ${txt(right)})`;
      return '';
    };

    // Recto (odd): group on the right (outside edge)
    // Verso (even): group on the left (outside edge)
    lines.push(`    if calc.odd(n) {`);
    lines.push(`      ${buildRow(rectoLeft, rectoRight, 'right')}`);
    lines.push(`    } else {`);
    lines.push(`      ${buildRow(versoLeft, versoRight, 'left')}`);
    lines.push(`    }`);
  }

  return lines.join('\n');
}

function generateTextSetup(config: PdfBookConfig): string {
  const lines: string[] = [];
  const leading = (config.lineHeight - 1) * config.fontSize;

  lines.push(`#set text(size: ${config.fontSize}pt, lang: "en", hyphenate: false)`);
  lines.push(`#set par(justify: true, first-line-indent: ${config.paragraphIndent}em, leading: ${leading.toFixed(2)}pt)`);

  // Style level-1 headings (chapter titles) — no numbering, bold, larger
  const titleSize = (config.fontSize * 1.8).toFixed(1);
  lines.push(`#set heading(numbering: none)`);
  lines.push(`#show heading.where(level: 1): set text(size: ${titleSize}pt, weight: "bold")`);

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

  // Chapter title as level-1 heading (picked up by #outline)
  // Label with <book-chapter> so headers can distinguish from outline's "Contents" heading
  parts.push(`#heading(level: 1)[${escapeTypst(title)}] <book-chapter>`);
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
