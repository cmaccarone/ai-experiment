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
}

interface BookConfig {
  trimSize: { width: number; height: number }  // in points (1/72 inch)
  margins: {
    top: number
    bottom: number
    inside: number   // spine side
    outside: number
  }
  gutter: number | GutterLookup
  fonts: FontConfig
  theme?: ThemeConfig
}

// Gutter convergence lives in the engine.
// Caller provides either a fixed number or a lookup.
type GutterLookup = (pageCount: number) => number
```

When `gutter` is a `GutterLookup` function, the engine:
1. Lays out with gutter = 0
2. Gets page count, calls the lookup to get the real gutter
3. Re-lays out with the new gutter
4. Repeats until page count stabilizes (max 5 iterations, then uses last result)

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
- `<img>` — Images
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
  | { type: 'image'; src: string; alt?: string }
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

The engine uses fontkit (or equivalent) to:
- Parse font metrics (glyph widths, ascent, descent, line gap)
- Measure text for line breaking
- Subset fonts for PDF embedding (only include used glyphs)
- Extract kerning pairs

---

## Layout Engine

### Line Breaking — Knuth-Plass

The engine uses the Knuth-Plass algorithm for paragraph line breaking:
- Considers all possible break points simultaneously
- Minimizes total "badness" (uneven spacing) across the paragraph
- Produces justified text without rivers of white space
- Integrates with hyphenation for more break opportunities

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
  lines: number        // how many lines the drop cap spans (default: 3)
  fontFamily?: string  // optional decorative font override
  bold?: boolean       // default: true
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
    h1?: { fontSize?: number; bold?: boolean; align?: string }
    h2?: { fontSize?: number; bold?: boolean; align?: string }
    h3?: { fontSize?: number; bold?: boolean; align?: string }
    h4?: { fontSize?: number; bold?: boolean; align?: string }
    h5?: { fontSize?: number; bold?: boolean; align?: string }
    h6?: { fontSize?: number; bold?: boolean; align?: string }
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
    backgroundColor?: string
    padding?: number
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

Progress reporting via `postMessage`:
- `{ type: 'progress', phase: 'parsing', percent: 20 }`
- `{ type: 'progress', phase: 'layout', percent: 50 }`
- `{ type: 'progress', phase: 'rendering', percent: 80 }`
- `{ type: 'done', pdf: Uint8Array }`

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
- **fontkit** — Font parsing, metrics, subsetting (also a pdf-lib dependency)
- **htmlparser2** — HTML parsing
- **hyphen** (or similar) — Hyphenation patterns

No native dependencies. Fully browser-compatible.

---

## Performance Considerations

- **Web worker** — All heavy computation off the main thread
- **Font subsetting** — Only embed used glyphs, reducing PDF size
- **Streaming layout** — Process chapters sequentially to limit memory usage
- **Transferable buffers** — Use `Transferable` for the PDF `ArrayBuffer` when posting back to main thread
- **Gutter convergence** — Capped iterations prevent infinite loops
