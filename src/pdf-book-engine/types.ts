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

// ── Page Context ──

export interface PageContext {
  pageNumber: number;
  isRecto: boolean;
  isChapterOpener: boolean;
  chapterTitle: string;
  bookTitle: string;
  totalPages: number;
}

// ── Header / Footer ──

export interface HeaderFooterContent {
  left?: string;
  center?: string;
  right?: string;
  font?: 'body' | 'bodyItalic' | 'heading';
  fontSize?: number;
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
  header?: (ctx: PageContext) => HeaderFooterContent | null;
  footer?: (ctx: PageContext) => HeaderFooterContent | null;
  widowLines: number;
  orphanLines: number;
}

// ── Layout intermediates ──

export type FontStyle = 'body' | 'bodyItalic' | 'bodyBold' | 'heading';

export function getFontStyle(bold: boolean, italic: boolean): FontStyle {
  if (bold) return 'bodyBold';
  if (italic) return 'bodyItalic';
  return 'body';
}

// A measured word ready for line breaking
export interface MeasuredWord {
  text: string;
  width: number;
  fontStyle: FontStyle;
  link?: string;
}

// Result of line breaking: a single typeset line
export interface TypesetLine {
  words: MeasuredWord[];
  width: number;           // actual content width
  availableWidth: number;  // target width
  isLastLine: boolean;     // don't justify last line of paragraph
}

// A laid-out line with vertical position info
export interface LayoutLine {
  typesetLine: TypesetLine;
  y: number;  // y position from top of page (points)
  x: number;  // x position (left margin, accounts for gutter)
}

// Footnote reference
export interface Footnote {
  index: number;
  url: string;
}

// A fully laid-out page
export interface LayoutPage {
  pageNumber: number;
  isRecto: boolean;
  isChapterOpener: boolean;
  chapterTitle: string;
  lines: LayoutLine[];
  footnotes: Footnote[];
  images: LayoutImage[];
}

export interface LayoutImage {
  src: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

// Validation warning
export interface ValidationWarning {
  code: string;
  message: string;
}
