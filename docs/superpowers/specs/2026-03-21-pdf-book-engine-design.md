# PDF Book Rendering Engine — Design Spec

## Overview

A general-purpose, TypeScript-based PDF book rendering engine that takes HTML content and configuration, and produces print-ready PDF output. Runs in a web worker for non-blocking operation. Uses pdf-lib for PDF generation.

The engine is **not** tied to any specific print service. Printer-specific details (margins, gutter tables) are provided by the caller.

---

## Core API

```typescript
interface BookEngine {
  render(content: Chapter[], config: BookConfig): Promise<Uint8Array>
}

interface Chapter {
  title: string
  html: string       // HTML content (from TipTap or any source)
  startOnRecto?: boolean  // default: true
  pageNumbering?: 'roman' | 'arabic' | 'none'  // default: 'arabic'
  images?: Record<string, ArrayBuffer>  // map of src URL → image bytes (pre-resolved by caller)
}

interface BookConfig {
  bookTitle?: string   // used for running headers when configured as 'bookTitle'
  trimSize: { width: number; height: number }  // in points (1/72 inch)
  margins: {
    top: number
    bottom: number
    inside: number   // spine side
    outside: number
  }
  gutter: number | GutterTable
  fonts: FontConfig
  theme?: ThemeConfig
}

// Gutter convergence lives in the engine.
// Caller provides either a fixed number or a serializable lookup table.
// GutterTable maps a max page count to a gutter value in points.
// E.g., { 100: 18, 200: 24, 300: 30, Infinity: 36 }
// The engine finds the smallest key >= actual page count.
type GutterTable = Record<number, number>
```

When `gutter` is a `GutterTable`, the engine:
1. Lays out with gutter = 0
2. Gets page count, looks up the gutter value from the table (smallest key >= page count)
3. Re-lays out with the new gutter
4. Repeats until page count stabilizes (max 5 iterations, then uses last result)

`GutterTable` is a plain object, so it survives `postMessage` serialization to web workers.

---

## Pipeline

```
HTML string (per chapter)
  → Parse (htmlparser2)
  → Block/Inline tree
  → Layout engine (line breaking, page breaking)
  → pdf-lib rendering
  → PDF bytes (Uint8Array)
```

---

## HTML Parsing

The engine accepts standard HTML as produced by rich text editors like TipTap. It parses HTML into an intermediate block/inline tree, then lays that out into pages.

### Supported Elements

**Block-level:**
- `<h1>` – `<h6>` — Headings, mapped to font sizes/weights via theme
- `<p>` — Paragraphs
- `<blockquote>` — Block quotes (indented, optional italic)
- `<ul>`, `<ol>`, `<li>` — Lists with bullets/numbers, proper indentation
- `<pre>`, `<code>` (block) — Code blocks in monospace font
- `<img>` — Images (block-level only)
- `<figure>`, `<figcaption>` — Image with caption
- `<hr>` — Section breaks / ornamental dividers
- `<table>`, `<tr>`, `<td>`, `<th>` — Basic tables

**Inline:**
- `<strong>`, `<b>` — Bold
- `<em>`, `<i>` — Italic
- `<u>` — Underline
- `<code>` (inline) — Inline monospace
- `<a>` — Links (rendered as styled text, not clickable in print)
- `<sup>`, `<sub>` — Superscript/subscript
- `<br>` — Line break

**Ignored:**
- `<script>`, `<style>`, CSS classes, arbitrary `style` attributes
- The engine renders *semantic* HTML, not arbitrary CSS

### Intermediate Representation

```typescript
type Block =
  | { type: 'heading'; level: 1 | 2 | 3 | 4 | 5 | 6; runs: InlineRun[] }
  | { type: 'paragraph'; runs: InlineRun[] }
  | { type: 'blockquote'; children: Block[] }
  | { type: 'list'; ordered: boolean; items: Block[][] }
  | { type: 'codeBlock'; text: string }
  | { type: 'figure'; src: string; alt?: string; caption?: InlineRun[] }  // src is a key into Chapter.images
  | { type: 'table'; rows: TableRow[] }
  | { type: 'horizontalRule' }

interface InlineRun {
  text: string
  bold?: boolean
  italic?: boolean
  underline?: boolean
  code?: boolean
  superscript?: boolean
  subscript?: boolean
  link?: string
}

interface TableRow {
  cells: { blocks: Block[]; header?: boolean }[]
}
```

---

## Fonts

The engine does **not** bundle any fonts. The caller provides font files (OTF/TTF) as `ArrayBuffer`s.

```typescript
interface FontConfig {
  body: FontFamily
  heading?: FontFamily    // defaults to body
  monospace?: FontFamily  // defaults to body
}

interface FontFamily {
  regular: ArrayBuffer    // font file bytes
  bold?: ArrayBuffer
  italic?: ArrayBuffer
  boldItalic?: ArrayBuffer
}
```

Font handling is split between two layers:

**fontkit** (for layout — text measurement):
- Parse font metrics (glyph widths, ascent, descent, line gap)
- Measure text for line breaking
- Extract kerning pairs

**pdf-lib** (for rendering — PDF embedding):
- Embed fonts into the PDF via `embedFont()` with `subset: true`
- pdf-lib handles font subsetting internally; the engine does NOT manually subset

Note: The standard `pdf-lib` + `@pdf-lib/fontkit` pairing has known compatibility issues with modern bundlers. If integration problems arise, use the maintained `@pdfme/pdf-lib` fork which bundles fontkit directly.

---

## Layout Engine

### Line Breaking — Knuth-Plass

The engine uses the Knuth-Plass algorithm for paragraph line breaking:
- Considers all possible break points simultaneously
- Minimizes total "badness" (uneven spacing) across the paragraph
- Produces justified text without rivers of white space
- Integrates with hyphenation for more break opportunities

Implementation approach: Use the existing `tex-linebreak` npm package (by Robert Knight) as a starting point rather than implementing from scratch. This is a well-tested JS implementation of the algorithm. If it needs modification for our use case, fork and adapt.

### Hyphenation

Uses a hyphenation library (e.g., `hyphen`) with language-specific patterns. Hyphenation points provide additional break opportunities for the Knuth-Plass algorithm.

### Page Breaking

Vertical page breaking with these constraints:
- **Widow control** — No single line of a paragraph at the top of a page
- **Orphan control** — No single line of a paragraph at the bottom of a page
- **Keep-with-next** — Headings stay with the following content
- **Keep-together** — Code blocks and short elements don't split across pages
- **Chapter starts** — Chapters start on recto (right-hand, odd) pages by default. Blank verso pages are inserted as needed.

### Paragraph Formatting

- **Justified text** by default (configurable)
- **First paragraph** after a heading: no indent
- **Subsequent paragraphs**: first-line indent, no space between
- **Block quotes**: indented from both sides

---

## Page Structure

### Page Regions

Each page has:
- **Content area** — Main text, bounded by margins + gutter
- **Header area** — Running header in the top margin
- **Footer area** — Page number in the bottom margin

### Running Headers

- **Verso (left) pages**: Book title or chapter title (configurable)
- **Recto (right) pages**: Chapter title or section title (configurable)
- **Suppressed on**: Chapter opener pages, blank pages

### Page Numbers

- **Position**: Bottom-outside (left on verso, right on recto) by default
- **Front matter**: Roman numerals (i, ii, iii, iv...)
- **Body**: Arabic numerals (1, 2, 3...)
- **Suppressed on**: Blank pages, optionally on chapter openers

### Recto/Verso Awareness

- Inside margin (spine side) is wider to account for binding
- Gutter is added to the inside margin
- Page numbers and running headers flip sides on recto vs verso

---

## Chapter Openers

The first page of each chapter has special treatment:
- Extra whitespace at the top (configurable, e.g., 1/3 page drop)
- Chapter title rendered with heading style
- No running header
- Optional drop cap on first paragraph

---

## Drop Caps

Optional feature, configured per-theme:

```typescript
interface DropCapConfig {
  enabled: boolean
  lines: number           // how many lines the drop cap spans (default: 3)
  font?: ArrayBuffer      // optional decorative font override (OTF/TTF bytes)
  bold?: boolean          // default: true
}
```

Implementation:
1. Extract the first letter from the first paragraph of the chapter
2. Render it at a size that spans `lines` number of body text lines
3. Wrap body text around it with a left indent for those lines
4. Remaining text flows full-width

---

## Theme Configuration

```typescript
interface ThemeConfig {
  body?: {
    fontSize?: number       // in points, default: 11
    leading?: number        // line height in points, default: 14
    textAlign?: 'justify' | 'left' | 'right' | 'center'
    firstLineIndent?: number // in points, default: 20
  }
  headings?: {
    h1?: { fontSize?: number; bold?: boolean; align?: 'left' | 'center' | 'right' }
    h2?: { fontSize?: number; bold?: boolean; align?: 'left' | 'center' | 'right' }
    h3?: { fontSize?: number; bold?: boolean; align?: 'left' | 'center' | 'right' }
    h4?: { fontSize?: number; bold?: boolean; align?: 'left' | 'center' | 'right' }
    h5?: { fontSize?: number; bold?: boolean; align?: 'left' | 'center' | 'right' }
    h6?: { fontSize?: number; bold?: boolean; align?: 'left' | 'center' | 'right' }
  }
  chapterOpener?: {
    topDrop?: number         // fraction of page height for top spacing (default: 0.33)
    titleAlign?: 'left' | 'center' | 'right'
  }
  dropCap?: DropCapConfig
  blockquote?: {
    indent?: number
    italic?: boolean
    fontSize?: number
  }
  codeBlock?: {
    fontSize?: number
    backgroundColor?: { r: number; g: number; b: number }  // RGB 0-1
    padding?: number
  }
  figure?: {
    captionFontSize?: number    // default: 9pt
    captionAlign?: 'left' | 'center' | 'right'  // default: 'center'
    captionItalic?: boolean     // default: true
    spacing?: number            // space between image and caption, default: 4pt
    marginTop?: number          // space above figure, default: 12pt
    marginBottom?: number       // space below figure, default: 12pt
  }
  runningHeader?: {
    verso?: 'bookTitle' | 'chapterTitle'  // default: chapterTitle
    recto?: 'chapterTitle' | 'sectionTitle' // default: sectionTitle
    fontSize?: number
    smallCaps?: boolean
  }
  pageNumber?: {
    position?: 'bottom-outside' | 'bottom-center'
    fontSize?: number
  }
  sectionBreak?: {
    style?: 'space' | 'asterisks' | 'line'
    spacing?: number
  }
}
```

---

## Image Handling

### Image Data

Images are **not** fetched by the engine. The caller pre-resolves all images and provides them as `ArrayBuffer`s in the `Chapter.images` map, keyed by the `src` attribute from the HTML.

```typescript
const chapter: Chapter = {
  title: 'Chapter 1',
  html: `
    <p>The cathedral rose above the city.</p>
    <figure>
      <img src="cathedral.jpg" alt="Notre-Dame at sunset">
      <figcaption>Notre-Dame de Paris, viewed from the Seine.</figcaption>
    </figure>
    <p>Its spire had been rebuilt after the fire.</p>
  `,
  images: {
    'cathedral.jpg': cathedralArrayBuffer  // pre-fetched by caller
  }
}
```

A bare `<img>` (without `<figure>`) is treated as a figure with no caption.

### Captions

Captions are rendered below the image in a smaller font size (configurable via theme), centered by default. The caption is part of the figure block — it stays with the image and never separates across a page break.

### Image Placement — Single-Pass with Backfill

The core goal: **no blank gaps**. When an image can't fit on the current page, text from after the image pulls up to fill the rest of the page. The image goes on the next page. No wasted space.

Images are placed using a **single-pass, document-order** algorithm. This avoids re-layout iteration and O(n²) blowup:

1. The layout engine processes blocks in document order (paragraphs, headings, figures, etc.)
2. When a figure is encountered, the engine checks: does it fit on the current page (image + caption)?
3. **If yes**: place it inline, continue with the next block
4. **If no**: defer the figure. **Pull text forward** from after the figure to fill the remainder of the current page. The deferred figure then goes at the **top of the next page**, followed by whatever text remains.

#### Example

Document order: `[para A] [para B] [FIGURE] [para C] [para D] [para E]`

The figure doesn't fit after para B. Without backfill you'd get:

```
Page 1:  para A, para B, ~~~blank space~~~   ← bad
Page 2:  FIGURE, para C, para D, para E
```

With backfill, para C (and as much of D as fits) pull up:

```
Page 1:  para A, para B, para C, para D...   ← full page, no gap
Page 2:  FIGURE, ...para D cont, para E
```

The figure stays within a page of where it was referenced, and no page has a blank hole.

#### Rules

- **No blank gaps.** When a figure is deferred, subsequent text always backfills the current page. The engine never leaves empty space where the figure was supposed to go.
- **At most one deferred figure at a time.** If the engine encounters a second figure while one is already deferred, it forces a page break, places the first deferred figure, then evaluates the second.
- **No re-injection.** A figure is placed exactly once. It either goes where it appears in the flow, or it moves to the top of the next page. It never bounces further.
- **Backfill is bounded.** Only text blocks between the deferred figure and the next figure (or end of chapter) are candidates for backfill. This keeps the image within a page or two of its reference point.
- **Chapter boundaries reset.** Deferred figures are flushed before a chapter ends — they never leak into the next chapter.
- **Full-page images.** If a figure (image + caption) fills an entire page on its own, that's fine — it becomes a dedicated image page. Text before it fills the previous page, text after it starts the next page.

This gives "close to where referenced" placement without complex float algorithms or iterative re-layout.

### Image Sizing

- Images are scaled to fit within the content area width, maintaining aspect ratio
- If the image is smaller than the content area, it is centered horizontally at its natural size
- Maximum image height: 70% of the content area height (to leave room for surrounding text and caption)
- If an image `src` is not found in the `images` map, render a placeholder box with the alt text

### Theme Configuration for Images

```typescript
// Added to ThemeConfig
figure?: {
  captionFontSize?: number    // default: 9pt
  captionAlign?: 'left' | 'center' | 'right'  // default: 'center'
  captionItalic?: boolean     // default: true
  spacing?: number            // space between image and caption, default: 4pt
  marginTop?: number          // space above figure, default: 12pt
  marginBottom?: number       // space below figure, default: 12pt
}
```

---

## Table Layout

Tables use a simple layout algorithm (not the full CSS table model):

- **Column widths**: Proportional to content. Measure the natural width of each column's content, then distribute available width proportionally. Columns have a minimum width to prevent collapse.
- **Cell padding**: Configurable via theme (default: 4pt)
- **Borders**: Simple 0.5pt lines between cells (configurable: all, header-only, none)
- **Text wrapping**: Cell content wraps within the calculated column width
- **Page spanning**: Tables do NOT span pages. If a table doesn't fit on the current page, it moves to the next page. If a table is taller than a full page, it is truncated with a warning.
- **Header rows**: Optional, rendered in bold

This is intentionally simple. Complex table layout (merged cells, column spans) is out of scope.

---

## Section Tracking for Running Headers

The engine tracks the current "section" for running headers by watching for `<h2>` elements during layout. When a `<h2>` is encountered, it becomes the current section title for subsequent recto page headers until the next `<h2>` or chapter boundary.

- `<h1>` = chapter-level (used for chapter title headers)
- `<h2>` = section-level (used for section title headers)
- Deeper headings (`<h3>`–`<h6>`) do not affect running headers

---

## What the Engine Does NOT Do

- **No TOC generation** — The caller builds TOC HTML if needed and passes it as a chapter
- **No front/back matter knowledge** — The engine renders chapters in order. The caller decides what's front matter vs body matter and sets page numbering accordingly.
- **No printer-specific logic** — Margins, gutter, trim size all come from the caller
- **No font bundling** — Caller provides font file bytes
- **No Markdown parsing** — Input is HTML
- **No file I/O** — Runs in a web worker, receives data in memory, returns PDF bytes

---

## Web Worker Architecture

The engine runs in a web worker to avoid blocking the main thread.

```typescript
// Main thread
const worker = new Worker('book-engine-worker.js')
worker.postMessage({ content: chapters, config: bookConfig })
worker.onmessage = (e) => {
  const pdfBytes: Uint8Array = e.data.pdf
  // download or display
}

// Worker thread
self.onmessage = async (e) => {
  const { content, config } = e.data
  const engine = new BookEngine()
  const pdf = await engine.render(content, config)
  self.postMessage({ pdf }, [pdf.buffer])
}
```

Worker message protocol (discriminated union):

```typescript
type WorkerMessage =
  | { type: 'progress'; phase: 'parsing' | 'layout' | 'rendering'; percent: number }
  | { type: 'done'; pdf: Uint8Array }
  | { type: 'error'; message: string; details?: string }
  | { type: 'warning'; message: string }  // e.g., missing image, truncated table
```

Note: All data crossing the worker boundary must be serializable via the structured clone algorithm. This is why `GutterTable` is a plain object (not a function), fonts are `ArrayBuffer`s, and images are pre-resolved `ArrayBuffer`s.

---

## Error Handling

- **Invalid HTML**: Parse gracefully, skip unrecognized elements, render what's understood
- **Missing fonts**: Error if required `regular` font is not provided. Warn if bold/italic variants are missing (fall back to regular with synthetic styling where possible).
- **Images**: If an image `src` can't be loaded, render a placeholder box with alt text
- **Gutter convergence**: Cap at 5 iterations, use last stable result
- **Oversized content**: If a single element (e.g., image) exceeds the content area, scale it down to fit

---

## Package Structure

```
pdf-book-engine/
├── src/
│   ├── index.ts              # Public API exports
│   ├── engine.ts             # Main BookEngine orchestrator
│   ├── parser/
│   │   └── html.ts           # HTML → Block/Inline tree
│   ├── layout/
│   │   ├── line-breaking.ts  # Knuth-Plass algorithm
│   │   ├── page-breaking.ts  # Vertical page breaking
│   │   └── layout.ts         # Block layout orchestrator
│   ├── renderer/
│   │   └── pdf.ts            # Block tree → pdf-lib → PDF bytes
│   ├── fonts/
│   │   └── manager.ts        # Font loading, metrics, subsetting
│   ├── types.ts              # All shared type definitions
│   └── worker.ts             # Web worker entry point
├── package.json
├── tsconfig.json
└── README.md
```

---

## Dependencies

- **pdf-lib** — PDF generation
- **fontkit** — Font parsing and metrics (used by pdf-lib internally for embedding; used directly for text measurement during layout)
- **htmlparser2** — HTML parsing
- **hyphen** (or similar) — Hyphenation patterns
- **tex-linebreak** — Knuth-Plass line breaking algorithm

No native dependencies. Fully browser-compatible.

Note: If `pdf-lib` + `@pdf-lib/fontkit` integration is problematic with bundlers, switch to `@pdfme/pdf-lib` (maintained fork with built-in fontkit support).

---

## Performance Considerations

- **Web worker** — All heavy computation off the main thread
- **Font subsetting** — pdf-lib's `embedFont({ subset: true })` only embeds used glyphs, reducing PDF size
- **Streaming layout** — Process chapters sequentially to limit memory usage
- **Transferable buffers** — Use `Transferable` for the PDF `ArrayBuffer` when posting back to main thread
- **Gutter convergence** — Capped iterations prevent infinite loops
