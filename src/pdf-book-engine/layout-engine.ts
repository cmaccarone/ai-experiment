import type {
  Block, Chapter, PdfBookConfig, FontStyle, MeasuredWord,
  TypesetLine, LayoutLine, LayoutPage, LayoutImage, Footnote, StyledRun,
} from './types.js';
import { getFontStyle } from './types.js';
import { parseHtml } from './html-parser.js';
import { FontManager } from './font-manager.js';
import { TextMeasurer } from './text-measurer.js';
import { breakLines } from './line-breaker.js';
import { breakPages, type ChapterContent, type PageItem } from './page-breaker.js';
import { lookupGutter, LULU_PROFILE } from './printer-profiles.js';
import { inchesToPoints, emsToPoints, isRecto } from './utils.js';

export interface LayoutResult {
  pages: LayoutPage[];
  warnings: string[];
}

/**
 * Orchestrates the full layout pipeline:
 *   HTML → IR → measure → line break → page break → position
 *
 * Runs the adaptive gutter convergence loop.
 */
export async function layoutBook(
  chapters: Chapter[],
  config: PdfBookConfig,
  images: Record<string, ArrayBuffer>,
  fontManager: FontManager,
): Promise<LayoutResult> {
  const printer = config.printer ?? LULU_PROFILE;
  const warnings: string[] = [];

  // Parse all chapters into IR (done once, reused across convergence iterations)
  const chapterBlocks: Array<{ title: string; blocks: Block[] }> = chapters.map((ch) => ({
    title: ch.title,
    blocks: parseHtml(ch.html),
  }));

  const measurer = new TextMeasurer(fontManager, config.fontSize);

  // Page dimensions in points
  const trimW = inchesToPoints(config.trimWidth);
  const trimH = inchesToPoints(config.trimHeight);
  const marginTop = inchesToPoints(config.margins.top);
  const marginBottom = inchesToPoints(config.margins.bottom);
  const marginOutside = inchesToPoints(config.margins.outside);

  const lineHeight = config.fontSize * config.lineHeight;
  const paragraphIndent = emsToPoints(config.paragraphIndent, config.fontSize);

  // Heading sizes
  const h2Size = config.fontSize * 1.4;
  const h3Size = config.fontSize * 1.2;

  // Convergence loop
  let estimatedPages = chapters.length * 20; // rough initial estimate
  let gutter = lookupGutter(printer, estimatedPages);
  let pages: LayoutPage[] = [];

  for (let iteration = 0; iteration < 5; iteration++) {
    const marginInside = inchesToPoints(gutter);
    const textBlockWidth = trimW - marginInside - marginOutside;
    const availableHeight = trimH - marginTop - marginBottom;

    // Measure and break all chapters
    const chapterContents: ChapterContent[] = [];

    let globalFootnoteIndex = 0;

    for (const { title, blocks } of chapterBlocks) {
      const items: PageItem[] = [];
      const chapterFootnotes: Footnote[] = [];

      // Chapter top drop
      if (config.chapterTopDrop > 0) {
        items.push({
          type: 'blankSpace',
          height: inchesToPoints(config.chapterTopDrop),
        });
      }

      // Chapter title
      const headingFontStyle: FontStyle = fontManager.hasFont('heading') ? 'heading' : 'body';
      const titleFontSize = config.fontSize * 1.8;
      items.push({
        type: 'chapterTitle',
        height: titleFontSize * config.lineHeight * 2, // title + space after
        text: title,
      });

      // Process each block
      for (const block of blocks) {
        switch (block.type) {
          case 'paragraph': {
            const lineItems = breakParagraph(
              block.runs, measurer, textBlockWidth, paragraphIndent,
              lineHeight, config.fontSize, chapterFootnotes, globalFootnoteIndex,
            );
            globalFootnoteIndex += countLinks(block.runs);
            items.push(...lineItems);
            // Paragraph spacing
            items.push({ type: 'blankSpace', height: lineHeight * 0.5 });
            break;
          }

          case 'heading': {
            const hSize = block.level === 2 ? h2Size : h3Size;
            const hHeight = hSize * config.lineHeight;
            // Space before heading
            items.push({ type: 'blankSpace', height: hHeight * 0.8 });
            items.push({
              type: 'heading',
              height: hHeight,
              text: block.runs.map((r) => r.text).join(''),
              level: block.level,
              keepWithNext: true,
            });
            items.push({ type: 'blankSpace', height: hHeight * 0.3 });
            break;
          }

          case 'list': {
            for (let idx = 0; idx < block.items.length; idx++) {
              const itemRuns = block.items[idx];
              const bullet = block.ordered ? `${idx + 1}. ` : '• ';
              const bulletWidth = measurer.measureText(bullet, 'body');
              const itemWidth = textBlockWidth - bulletWidth;

              // Prepend bullet to first run
              const bulletRun: StyledRun = { text: bullet, bold: false, italic: false };
              const allRuns = [bulletRun, ...itemRuns];
              const measuredWords = measurer.measureRuns(allRuns);
              const lines = breakLines(measuredWords, textBlockWidth, measurer.measureSpace('body'));

              for (const line of lines) {
                items.push({
                  type: 'line',
                  height: lineHeight,
                  typesetLine: line,
                });
              }
            }
            items.push({ type: 'blankSpace', height: lineHeight * 0.5 });
            break;
          }

          case 'image': {
            const imgBuffer = images[block.src];
            if (imgBuffer) {
              // We'll determine actual dimensions during PDF rendering
              // For layout, assume a proportional height placeholder
              const imgHeight = textBlockWidth * 0.6; // rough aspect ratio
              items.push({
                type: 'image',
                height: imgHeight + lineHeight, // image + spacing
                imageSrc: block.src,
                imageWidth: textBlockWidth,
                imageHeight: imgHeight,
              });
            }
            break;
          }

          case 'lineBreak': {
            items.push({ type: 'blankSpace', height: lineHeight });
            break;
          }
        }
      }

      chapterContents.push({
        title,
        forceRecto: config.chapterStartRecto,
        items,
      });
    }

    // Page breaking
    const pageBreaks = breakPages(chapterContents, {
      availableHeight,
      widowLines: config.widowLines,
      orphanLines: config.orphanLines,
    });

    // Convert to LayoutPages with positioned elements
    pages = pageBreaks.map((pb, idx) => {
      const pageNum = idx + 1;
      const recto = isRecto(pageNum);
      const marginLeft = recto ? inchesToPoints(gutter) : marginOutside;

      let y = marginTop;
      const lines: LayoutLine[] = [];
      const pageImages: LayoutImage[] = [];
      const pageFootnotes: Footnote[] = [];

      for (const item of pb.items) {
        switch (item.type) {
          case 'line':
            if (item.typesetLine) {
              lines.push({
                typesetLine: item.typesetLine,
                x: marginLeft + (item.indent ?? 0),
                y,
              });
            }
            y += item.height;
            break;

          case 'chapterTitle':
          case 'heading':
            // Rendered as a special line
            lines.push({
              typesetLine: {
                words: [{ text: item.text ?? '', width: 0, fontStyle: item.type === 'chapterTitle' ? 'heading' : 'body' }],
                width: 0,
                availableWidth: textBlockWidth,
                isLastLine: true,
              },
              x: marginLeft,
              y,
            });
            y += item.height;
            break;

          case 'image':
            if (item.imageSrc) {
              pageImages.push({
                src: item.imageSrc,
                x: marginLeft,
                y,
                width: item.imageWidth ?? textBlockWidth,
                height: item.imageHeight ?? 100,
              });
            }
            y += item.height;
            break;

          case 'blankSpace':
            y += item.height;
            break;
        }
      }

      return {
        pageNumber: pageNum,
        isRecto: recto,
        isChapterOpener: pb.isChapterOpener,
        chapterTitle: pb.chapterTitle,
        lines,
        footnotes: pageFootnotes,
        images: pageImages,
      };
    });

    // Check convergence
    const newGutter = lookupGutter(printer, pages.length);
    if (newGutter === gutter) {
      break; // Converged
    }
    gutter = newGutter;
  }

  return { pages, warnings };
}

function breakParagraph(
  runs: StyledRun[],
  measurer: TextMeasurer,
  textBlockWidth: number,
  paragraphIndent: number,
  lineHeight: number,
  fontSize: number,
  footnotes: Footnote[],
  footnoteStartIndex: number,
): PageItem[] {
  // Collect footnotes from links
  let fnIdx = footnoteStartIndex;
  for (const run of runs) {
    if (run.link) {
      fnIdx++;
      footnotes.push({ index: fnIdx, url: run.link });
    }
  }

  const measuredWords = measurer.measureRuns(runs);
  const spaceWidth = measurer.measureSpace('body');

  // First line is indented — pass the narrower first-line width so the line
  // breaker packs words correctly for both the indented first line and the
  // full-width subsequent lines.
  const firstLineWidth = textBlockWidth - paragraphIndent;
  const lines = breakLines(measuredWords, textBlockWidth, spaceWidth, firstLineWidth);

  return lines.map((line, i) => ({
    type: 'line' as const,
    height: lineHeight,
    typesetLine: line,
    ...(i === 0 && paragraphIndent > 0 ? { indent: paragraphIndent } : {}),
  }));
}

function countLinks(runs: StyledRun[]): number {
  return runs.filter((r) => r.link).length;
}
