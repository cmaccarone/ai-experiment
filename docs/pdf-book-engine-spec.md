# PDF Book Engine — Technical Specification

## Overview

A TypeScript PDF book engine that takes structured chapter data and produces print-ready PDFs suitable for Lulu print-on-demand. The engine handles professional book typography including justified text, proper page layout, and adaptive gutter margins.

---

## API

### Entry Point

```typescript
async function generateBook(
  chapters: Chapter[],
  config: PdfBookConfig,
  images: Record<string, ArrayBuffer>  // keyed by <img src> value
): Promise<Uint8Array>
```

### Types

```typescript
interface Chapter {
  title: string;          // Chapter title (rendered as chapter opener)
  html: string;           // Chapter body HTML content
}

interface PageContext {
  pageNumber: number;
  isRecto: boolean;        // Right-hand (odd) page
  isChapterOpener: boolean;
  chapterTitle: string;
  bookTitle: string;
  totalPages: number;
}

interface HeaderFooterContent {
  left?: string;
  center?: string;
  right?: string;
  font?: 'body' | 'bodyItalic' | 'heading';
  fontSize?: number;
}

interface PdfBookConfig {
  // Page dimensions (inches)
  trimWidth: number;
  trimHeight: number;

  // Margins (inches)
  margins: {
    top: number;
    bottom: number;
    inside: number;    // Fallback if no gutterTable provided
    outside: number;
  };

  // Adaptive gutter (optional — Lulu defaults used if omitted)
  gutterTable?: Array<{
    maxPages: number;  // Up to this many pages...
    gutter: number;    // ...use this gutter (inches)
  }>;

  // Fonts — user must provide (paths or ArrayBuffers)
  fonts: {
    body: string | ArrayBuffer;
    bodyItalic?: string | ArrayBuffer;
    bodyBold?: string | ArrayBuffer;
    heading?: string | ArrayBuffer;
  };

  // Typography
  fontSize: number;         // points
  lineHeight: number;       // multiplier (e.g., 1.4)
  paragraphIndent: number;  // ems

  // Chapter layout
  chapterStartRecto: boolean;  // Chapters start on right-hand page
  chapterTopDrop: number;      // Inches of blank space above chapter title

  // Headers & footers — render functions called per page
  header?: (ctx: PageContext) => HeaderFooterContent | null;
  footer?: (ctx: PageContext) => HeaderFooterContent | null;

  // Page break quality
  widowLines: number;   // Min lines at top of page (default 2)
  orphanLines: number;  // Min lines at bottom before break (default 2)
}
```

### Example Usage

```typescript
import { generateBook } from './pdf-book-engine';

const chapters: Chapter[] = [
  {
    title: "The Beginning",
    html: "<p>It was a dark and stormy night...</p>"
  },
  {
    title: "The Journey",
    html: "<p>They set off at dawn, carrying <em>nothing</em> but hope.</p>"
  }
];

const images: Record<string, ArrayBuffer> = {
  "map.png": mapImageBuffer,
};

const pdf = await generateBook(chapters, {
  trimWidth: 6,
  trimHeight: 9,
  margins: { top: 0.75, bottom: 0.75, inside: 0.75, outside: 0.5 },
  fonts: {
    body: bodyFontBuffer,
    bodyItalic: italicFontBuffer,
    bodyBold: boldFontBuffer,
    heading: headingFontBuffer,
  },
  fontSize: 11,
  lineHeight: 1.4,
  paragraphIndent: 1.5,
  chapterStartRecto: true,
  chapterTopDrop: 2,
  widowLines: 2,
  orphanLines: 2,

  header: (ctx) => {
    if (ctx.isChapterOpener) return null;
    return {
      left: ctx.isRecto ? undefined : ctx.bookTitle,
      right: ctx.isRecto ? ctx.chapterTitle : undefined,
      font: 'bodyItalic',
      fontSize: 9,
    };
  },

  footer: (ctx) => ({
    center: String(ctx.pageNumber),
    fontSize: 10,
  }),
}, images);
```

---

## Architecture

### Module Structure

```
src/pdf-book-engine/
├── index.ts              # Public API entry point (generateBook)
├── types.ts              # Shared types & interfaces
├── html-parser.ts        # HTML → block/inline IR
├── font-manager.ts       # Font loading, glyph metrics (via opentype.js)
├── text-measurer.ts      # Measures text runs using font metrics
├── line-breaker.ts       # Knuth-Plass line breaking algorithm
├── page-breaker.ts       # Assigns lines to pages (widow/orphan control)
├── layout-engine.ts      # Orchestrator: IR → positioned elements
├── pdf-writer.ts         # Low-level PDF object generation (via pdf-lib)
├── pdf-renderer.ts       # Renders positioned elements → PDF drawing ops
└── utils.ts              # Unit conversion, gutter table, helpers
```

### Pipeline

```
Chapter[]  →  HTML Parser  →  IR (block/inline tree)
                                    ↓
                              Text Measurer  ←  Font Manager (opentype.js)
                                    ↓
                              Line Breaker (Knuth-Plass)
                                    ↓
                              Page Breaker (widow/orphan)
                                    ↓
                              Layout Engine (coordinates)
                                    ↓
                              PDF Renderer  →  pdf-lib  →  Uint8Array
```

### Intermediate Representation (IR)

The HTML parser converts each chapter's HTML into a typed tree. All downstream stages work on this IR, not raw HTML.

```typescript
// Block-level nodes
type BlockNode =
  | { type: 'paragraph'; children: InlineNode[] }
  | { type: 'heading'; level: 2 | 3; children: InlineNode[] }
  | { type: 'list'; ordered: boolean; items: ListItem[] }
  | { type: 'image'; src: string; alt?: string }

type ListItem = { children: InlineNode[] }

// Inline nodes
type InlineNode =
  | { type: 'text-run'; text: string; style: InlineStyle }
  | { type: 'link'; href: string; children: InlineNode[] }

type InlineStyle = {
  italic: boolean;
  bold: boolean;
}
```

### Separation of Concerns

- **HTML Parser** → produces IR tree (knows HTML, nothing else)
- **Line Breaker** → sees flat text runs with widths, finds optimal breaks (knows Knuth-Plass, nothing about HTML or PDF)
- **Page Breaker** → sees "things with heights", assigns to pages (knows widow/orphan rules, nothing about text)
- **PDF Renderer** → gets "draw text at (x, y) in font F" instructions (knows PDF, nothing about layout logic)

Each stage has a narrow contract, enabling isolated unit testing.

---

## Key Design Decisions

### PDF Generation: Hybrid Approach

- **pdf-lib** handles structural plumbing: document creation, pages, font embedding, stream encoding, cross-reference tables
- **Custom code** handles all text positioning, line breaking, and layout math
- This gives full control over typography while avoiding low-level PDF spec bugs

### Font Handling

- **opentype.js** parses font files (TTF/OTF) to extract per-glyph widths, kerning pairs, ascent/descent metrics
- **pdf-lib** embeds fonts into the PDF with subsetting
- **No bundled fonts** — user must provide all font files
- Fonts can be provided as file paths (strings) or `ArrayBuffer`s

### Line Breaking: Knuth-Plass

- Models paragraphs as boxes (glyphs), glue (stretchable spaces), and penalties
- Finds globally optimal line breaks minimizing total "badness" across all lines
- **No hyphenation** — breaks at word boundaries only
- Supports looseness parameter for page-break optimization

### Adaptive Gutter (Convergence Loop)

Thicker books need wider inside margins because pages curve at the spine. The engine:

1. Estimates page count from content
2. Looks up gutter from tier table (built-in Lulu defaults or user-provided)
3. Lays out the entire book with that gutter
4. Checks actual page count → looks up gutter for that count
5. If gutter changed, re-layouts. Repeats until stable (typically 1-2 iterations)

**Built-in Lulu defaults** are used unless the user provides a custom `gutterTable`.

### Links → Footnotes

`<a>` tags are rendered as styled inline text with the URL collected as a numbered footnote at the bottom of the page.

### Images

- Provided via an image map: `Record<string, ArrayBuffer>` keyed by `<img src>` values
- Rendered as block-level elements, full text-block width, maintaining aspect ratio
- Placed inline in the content flow (not floated)

---

## Supported HTML Elements (MVP)

### Block Elements
| Element | Behavior |
|---------|----------|
| `<p>` | Paragraph — justified text, first-line indent |
| `<h2>` | Section heading (within chapter) |
| `<h3>` | Sub-section heading |
| `<ul>` | Unordered list |
| `<ol>` | Ordered list |
| `<li>` | List item |
| `<img>` | Image — block-level, full-width |

### Inline Elements
| Element | Behavior |
|---------|----------|
| `<em>` / `<i>` | Italic |
| `<strong>` / `<b>` | Bold |
| `<br>` | Line break |
| `<a>` | Link — rendered as text, URL as page footnote |

---

## Phasing

### Phase 1 — MVP
- HTML parsing (all elements listed above)
- Font loading & text measurement (opentype.js)
- Knuth-Plass line breaking (no hyphenation)
- Page breaking with widow/orphan control
- Paragraph formatting (justified, indents)
- Recto/verso margins
- **Adaptive gutter with convergence loop (Lulu defaults)**
- Page numbers (via footer render function)
- Chapter starts on recto
- Images (block-level, full-width)
- Lists (ordered & unordered)
- Links with URL footnotes
- Unit tests + snapshot tests

### Phase 2 — Polish
- Running headers
- Chapter openers (top drop, styled title formatting)
- Drop caps
- Block quotes
- Section breaks (`<hr>`)

### Phase 3 — Nice to Have
- Tables
- Code blocks with background
- Web worker wrapper & progress messages

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pdf-lib` | PDF document structure, font embedding, page creation |
| `opentype.js` | Font file parsing, glyph metrics, kerning |

---

## Testing Strategy

### Unit Tests
- **Line breaker**: Test with mock text measurements, verify break points
- **Page breaker**: Test with mock line heights, verify page assignments and widow/orphan handling
- **HTML parser**: Verify IR output for known HTML inputs
- **Font manager**: Test metric extraction with real font files
- **Gutter convergence**: Test with various page count scenarios

### Snapshot Tests
- Generate PDFs from known chapter inputs with fixed config
- Compare output against known-good reference PDFs
- Detect regressions in layout, positioning, or rendering
