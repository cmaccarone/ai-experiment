import { rgb, type PDFPage, type PDFImage } from 'pdf-lib';
import type { PdfWriter } from './pdf-writer.js';
import type {
  LayoutPage, LayoutLine, LayoutImage, Footnote,
  PdfBookConfig, PageContext, HeaderFooterContent, FontStyle,
} from './types.js';
import { inchesToPoints } from './utils.js';

/**
 * Renders positioned layout elements onto PDF pages.
 */
export class PdfRenderer {
  constructor(
    private writer: PdfWriter,
    private config: PdfBookConfig,
    private embeddedImages: Map<string, any>,
  ) {}

  renderPage(layoutPage: LayoutPage, bookTitle: string, totalPages: number): void {
    const trimW = inchesToPoints(this.config.trimWidth);
    const trimH = inchesToPoints(this.config.trimHeight);

    const page = this.writer.addPage(trimW, trimH);

    // Render text lines
    for (const line of layoutPage.lines) {
      this.renderLine(page, line, trimH);
    }

    // Render images
    for (const img of layoutPage.images) {
      this.renderImage(page, img, trimH);
    }

    // Render footnotes
    if (layoutPage.footnotes.length > 0) {
      this.renderFootnotes(page, layoutPage.footnotes, trimH);
    }

    // Render header
    if (this.config.header) {
      const ctx = this.makePageContext(layoutPage, bookTitle, totalPages);
      const headerContent = this.config.header(ctx);
      if (headerContent) {
        this.renderHeaderFooter(page, headerContent, trimH, 'header');
      }
    }

    // Render footer
    if (this.config.footer) {
      const ctx = this.makePageContext(layoutPage, bookTitle, totalPages);
      const footerContent = this.config.footer(ctx);
      if (footerContent) {
        this.renderHeaderFooter(page, footerContent, trimH, 'footer');
      }
    }
  }

  private renderLine(page: PDFPage, line: LayoutLine, pageHeight: number): void {
    const ts = line.typesetLine;
    const words = ts.words;
    if (words.length === 0) return;

    // Calculate justification spacing
    let x = line.x;
    const y = pageHeight - line.y - this.config.fontSize; // PDF y is from bottom

    // Check if this is a heading/title line (has a single word with fontStyle heading)
    const isHeading = words.length === 1 && (words[0].fontStyle === 'heading');
    const isChapterTitle = isHeading && words[0].width === 0;

    let fontSize = this.config.fontSize;
    let fontStyle: FontStyle = 'body';

    if (isChapterTitle) {
      fontSize = this.config.fontSize * 1.8;
      fontStyle = 'heading';
    } else if (isHeading) {
      fontSize = this.config.fontSize * 1.4;
      fontStyle = 'heading';
    }

    if (isHeading || isChapterTitle) {
      // Render as a single text draw
      const font = this.writer.getFont(fontStyle);
      page.drawText(words[0].text, {
        x,
        y,
        size: fontSize,
        font,
        color: rgb(0, 0, 0),
      });
      return;
    }

    // Justified text — distribute extra space between words
    if (!ts.isLastLine && words.length > 1) {
      // Count space-separated word groups
      const nonSpaceWords = words.filter(w => !/^\s+$/.test(w.text));
      const gaps = nonSpaceWords.length - 1;

      if (gaps > 0) {
        const contentWidth = words.reduce((w, word) => w + word.width, 0);
        const spaceWords = words.filter(w => /^\s+$/.test(w.text));
        const totalSpaceWidth = spaceWords.reduce((w, s) => w + s.width, 0);
        const nonSpaceContentWidth = contentWidth - totalSpaceWidth;
        const extraSpace = ts.availableWidth - nonSpaceContentWidth;
        const spacePerGap = extraSpace / gaps;

        for (const word of words) {
          if (/^\s+$/.test(word.text)) {
            x += spacePerGap;
            continue;
          }

          const font = this.writer.getFont(word.fontStyle);
          page.drawText(word.text, {
            x,
            y,
            size: fontSize,
            font,
            color: rgb(0, 0, 0),
          });
          x += word.width;
        }
        return;
      }
    }

    // Left-aligned (last line of paragraph or single-word line)
    for (const word of words) {
      if (/^\s+$/.test(word.text)) {
        x += word.width;
        continue;
      }

      const font = this.writer.getFont(word.fontStyle);
      page.drawText(word.text, {
        x,
        y,
        size: fontSize,
        font,
        color: rgb(0, 0, 0),
      });
      x += word.width;
    }
  }

  private renderImage(page: PDFPage, img: LayoutImage, pageHeight: number): void {
    const embeddedImg = this.embeddedImages.get(img.src);
    if (!embeddedImg) return;

    page.drawImage(embeddedImg, {
      x: img.x,
      y: pageHeight - img.y - img.height,
      width: img.width,
      height: img.height,
    });
  }

  private renderFootnotes(page: PDFPage, footnotes: Footnote[], pageHeight: number): void {
    const marginBottom = inchesToPoints(this.config.margins.bottom);
    const footnoteFontSize = this.config.fontSize * 0.8;
    const footnoteLineHeight = footnoteFontSize * 1.3;
    const marginLeft = inchesToPoints(this.config.margins.outside); // simplified

    let y = marginBottom + footnotes.length * footnoteLineHeight;

    // Draw separator line
    const font = this.writer.getFont('body');
    const sepY = pageHeight - (pageHeight - y - footnoteLineHeight);

    for (const fn of footnotes) {
      const text = `${fn.index}. ${fn.url}`;
      page.drawText(text, {
        x: marginLeft,
        y: y,
        size: footnoteFontSize,
        font,
        color: rgb(0.3, 0.3, 0.3),
      });
      y -= footnoteLineHeight;
    }
  }

  private renderHeaderFooter(
    page: PDFPage,
    content: HeaderFooterContent,
    pageHeight: number,
    position: 'header' | 'footer',
  ): void {
    const fontSize = content.fontSize ?? 9;
    const fontStyle: FontStyle = content.font ?? 'body';
    const font = this.writer.getFont(fontStyle);
    const trimW = inchesToPoints(this.config.trimWidth);
    const marginOutside = inchesToPoints(this.config.margins.outside);

    const y = position === 'header'
      ? pageHeight - inchesToPoints(this.config.margins.top) * 0.6
      : inchesToPoints(this.config.margins.bottom) * 0.4;

    if (content.left) {
      page.drawText(content.left, {
        x: marginOutside,
        y,
        size: fontSize,
        font,
        color: rgb(0.3, 0.3, 0.3),
      });
    }

    if (content.center) {
      const textWidth = font.widthOfTextAtSize(content.center, fontSize);
      page.drawText(content.center, {
        x: (trimW - textWidth) / 2,
        y,
        size: fontSize,
        font,
        color: rgb(0.3, 0.3, 0.3),
      });
    }

    if (content.right) {
      const textWidth = font.widthOfTextAtSize(content.right, fontSize);
      page.drawText(content.right, {
        x: trimW - marginOutside - textWidth,
        y,
        size: fontSize,
        font,
        color: rgb(0.3, 0.3, 0.3),
      });
    }
  }

  private makePageContext(page: LayoutPage, bookTitle: string, totalPages: number): PageContext {
    return {
      pageNumber: page.pageNumber,
      isRecto: page.isRecto,
      isChapterOpener: page.isChapterOpener,
      chapterTitle: page.chapterTitle,
      bookTitle,
      totalPages,
    };
  }
}
