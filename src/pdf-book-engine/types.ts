// ── Block & Run Model (flat, no IR tree) ──

export type StyledRun = {
  text: string;
  bold: boolean;
  italic: boolean;
  link?: string;
};

export type Block =
  | { type: 'paragraph'; runs: StyledRun[] }
  | { type: 'heading'; level: 2 | 3; runs: StyledRun[] }
  | { type: 'list'; ordered: boolean; items: StyledRun[][] }
  | { type: 'image'; src: string; alt?: string }
  | { type: 'lineBreak' };

// ── Chapter ──

export interface Chapter {
  title: string;
  html: string;
}

// ── Header / Footer ──

/**
 * Declarative header/footer layout for book pages.
 *
 * Use `outside` and `inside` for book-aware positioning:
 * - `outside` = right on recto (odd) pages, left on verso (even) pages
 * - `inside` = left on recto pages, right on verso pages
 *
 * Content strings support placeholders:
 * - `PAGE` → current page number
 * - `CHAPTER` → current chapter title
 *
 * Example: `{ outside: "PAGE", inside: "CHAPTER", separator: "|" }`
 * renders as `Chapter Title | 5` on recto, `5 | Chapter Title` on verso.
 */
export interface HeaderFooterConfig {
  outside?: string;
  inside?: string;
  center?: string;
  separator?: string;
  fontSize?: number;
  hideOnChapterOpener?: boolean;
}

// ── Printer Profile ──

export interface PrinterProfile {
  name: string;
  gutterTable: Array<{ maxPages: number; gutter: number }>;
  minMargins?: { top: number; bottom: number; inside: number; outside: number };
  allowedTrimSizes?: Array<{ width: number; height: number; label?: string }>;
  bleedRequired?: boolean;
  maxPageCount?: number;
}

// ── Config ──

export interface PdfBookConfig {
  printer?: PrinterProfile;
  trimWidth: number;
  trimHeight: number;
  margins: {
    top: number;
    bottom: number;
    inside: number;
    outside: number;
  };
  fonts: {
    body: string | ArrayBuffer;
    bodyItalic?: string | ArrayBuffer;
    bodyBold?: string | ArrayBuffer;
    heading?: string | ArrayBuffer;
  };
  fontSize: number;
  lineHeight: number;
  paragraphIndent: number;
  chapterStartRecto: boolean;
  chapterTopDrop: number;
  header?: HeaderFooterConfig;
  footer?: HeaderFooterConfig;
  widowLines: number;
  orphanLines: number;
  tableOfContents?: boolean;
}

// ── Font style identifiers ──

export type FontStyle = 'body' | 'bodyItalic' | 'bodyBold' | 'heading';

// Validation warning
export interface ValidationWarning {
  code: string;
  message: string;
}
