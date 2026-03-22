# Plan: Replace pdf-lib Pipeline with Typst Backend

## Summary

Replace the entire custom typesetting pipeline (opentype.js measurement → Knuth-Plass line breaker → page breaker → pdf-lib renderer) with **Typst**, a modern typesetting engine that handles all of this natively with superior quality.

**Key insight:** Typst replaces 7 of 11 source files. Instead of manually measuring fonts, breaking lines, breaking pages, and rendering to PDF, we generate Typst markup and let Typst's compiler handle everything — including Knuth-Plass line breaking, hyphenation, widow/orphan control, and PDF output.

## Architecture Change

### Before (current)
```
HTML → IR blocks → measure words → line break → page break → position → pdf-lib → PDF
        (7 files doing custom typesetting)
```

### After (Typst)
```
HTML → IR blocks → Typst markup → Typst compiler → PDF
        (2 new files)
```

## What We Keep (unchanged or minor edits)

| File | Status | Reason |
|------|--------|--------|
| `types.ts` | Keep (minor edits) | Public API types. Remove internal layout types (MeasuredWord, TypesetLine, LayoutLine, etc.) that are no longer needed. Keep Chapter, PdfBookConfig, PrinterProfile, etc. |
| `printer-profiles.ts` | Keep as-is | Lulu/KDP/Ingram gutter tables — used to compute inside margins in the Typst template |
| `validator.ts` | Keep as-is | Config validation still needed |
| `utils.ts` | Keep as-is | Unit conversions still useful for template generation |
| `html-parser.ts` | Keep as-is | HTML → IR blocks. The IR is reused as input to the Typst markup generator |

## What We Remove

| File | Reason |
|------|--------|
| `font-manager.ts` | Typst loads fonts directly from file paths or bytes |
| `text-measurer.ts` | Typst measures text internally |
| `line-breaker.ts` | Typst has Knuth-Plass built in (+ hyphenation, microtypography) |
| `page-breaker.ts` | Typst handles page breaking with native widow/orphan control |
| `layout-engine.ts` | Replaced by Typst markup generator |
| `pdf-renderer.ts` | Typst renders PDF natively |
| `pdf-writer.ts` | Typst generates PDF natively |

## What We Add

### 1. `typst-generator.ts` — Typst Markup Generator
Converts the IR blocks + config into a complete `.typ` document string.

Responsibilities:
- Generate page setup from PdfBookConfig (trim size, margins, gutter from printer profile convergence)
- Map font config to Typst `#set text(font: ...)` declarations
- Convert IR blocks (paragraphs, headings, lists, images) to Typst markup
- Generate header/footer rules using `#set page(header: ..., footer: ...)`
- Handle chapter structure with page breaks, recto starts, chapter top drops
- Preserve bold/italic/link styling from StyledRuns

### 2. `typst-compiler.ts` — Compiler Abstraction
Wraps Typst compilation with support for both Node.js and browser environments.

```typescript
interface TypstCompilerOptions {
  fontPaths?: string[];          // Node.js: filesystem paths to font directories
  fontBuffers?: Map<string, Uint8Array>; // Browser: font data as buffers
}

interface TypstCompiler {
  compile(source: string, options?: TypstCompilerOptions): Promise<Uint8Array>;
}
```

Node.js backend: Uses the `typst` npm package (CLI wrapper) or `@myriaddreamin/typst-ts-node-compiler` (native addon).
Browser backend: Uses `@anthropic-ai/typst-ts-web-compiler` (WASM) — can be added later.

### 3. Updated `index.ts` — Simplified Orchestration

```typescript
export async function generateBook(
  chapters: Chapter[],
  config: PdfBookConfig,
  images: Record<string, ArrayBuffer> = {},
): Promise<Uint8Array> {
  // 1. Validate config (same as before)
  // 2. Parse HTML chapters to IR blocks (same as before)
  // 3. Compute gutter from printer profile (simplified — use max estimate)
  // 4. Generate Typst markup from IR + config
  // 5. Compile Typst → PDF
  // 6. Return PDF bytes
}
```

## Gutter Convergence Simplification

The current code runs a convergence loop: estimate pages → compute gutter → layout → recount pages → repeat if gutter changed. With Typst, we can either:

**Option A (simple):** Use the largest gutter from the profile. For Lulu with a 300-page book, this means using 0.875" instead of the optimal 0.625". Slightly wider inside margin than necessary, but safe and eliminates the loop.

**Option B (accurate):** Compile twice. First pass with estimated gutter to get page count, then recompile with the correct gutter. Typst compiles fast enough (~100ms for a few hundred pages) that this is practical.

**Recommendation:** Option B — it's what the current code does, and Typst is fast enough to make it painless.

## Dependencies

### Remove
- `pdf-lib` (^1.17.1)
- `@pdf-lib/fontkit` (^1.1.1)
- `opentype.js` (^1.3.4)
- `@types/opentype.js` (^1.3.8)

### Add
- `typst` (npm package) — CLI wrapper for Node.js (~15-30MB binary)
  - OR `@myriaddreamin/typst-ts-node-compiler` — native Node.js addon (no subprocess)

## Font Handling

Current: Fonts passed as file paths or ArrayBuffers → loaded by opentype.js for measurement → re-embedded by pdf-lib for rendering.

With Typst:
- **File paths:** Pass directly to Typst via `--font-path` flag
- **ArrayBuffers:** Write to temp files and pass paths, OR use the native compiler API which accepts font bytes

The `PdfBookConfig.fonts` type stays the same (`string | ArrayBuffer`), but the implementation changes.

## Header/Footer Callbacks

Current: `config.header` and `config.footer` are JS functions `(ctx: PageContext) => HeaderFooterContent | null`. They're called per-page during rendering.

With Typst: Headers/footers are defined declaratively in the Typst template using `#set page(header: ..., footer: ...)`. Typst has built-in access to page number, total pages, and heading tracking.

**Migration:** Convert the callback pattern to Typst template rules. Most common patterns (page number in footer, chapter title in header, suppress on chapter openers) are native Typst features. The JS callback API changes to a declarative config:

```typescript
interface HeaderFooterConfig {
  style: 'book-standard';  // verso: book title, recto: chapter title, centered page number
  // OR custom Typst template string for advanced use
  customTemplate?: string;
}
```

## Test Updates

- `line-breaker.test.ts` → **Delete** (Typst owns line breaking)
- `page-breaker.test.ts` → **Delete** (Typst owns page breaking)
- `text-measurer.test.ts` → **Delete** (Typst owns measurement)
- `pdf-renderer.test.ts` → **Delete** (Typst owns rendering)
- `html-parser.test.ts` → **Keep** (parser unchanged)
- `validator.test.ts` → **Keep** (validator unchanged)
- `printer-profiles.test.ts` → **Keep** (profiles unchanged)
- `utils.test.ts` → **Keep** (utils unchanged)
- `integration.test.ts` → **Update** (test full pipeline with Typst)
- **New:** `typst-generator.test.ts` — test markup generation

## Implementation Steps

1. **Install Typst dependency** — `npm install typst` and verify it works
2. **Create `typst-generator.ts`** — the core new file that converts IR blocks to Typst markup
3. **Create `typst-compiler.ts`** — wraps Typst compilation
4. **Update `types.ts`** — remove unused internal layout types, update header/footer config
5. **Update `index.ts`** — rewire the pipeline: parse HTML → generate Typst → compile → return PDF
6. **Create `typst-generator.test.ts`** — test markup generation for all block types
7. **Update `integration.test.ts`** — test the full pipeline end-to-end
8. **Remove dead code** — delete the 7 replaced files and their tests
9. **Update `package.json`** — remove old deps, add Typst
10. **Run full test suite** and fix any issues

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Typst npm package is at RC version | Pin exact version; test thoroughly |
| Typst markup generation bugs | Comprehensive unit tests for each block type |
| Browser WASM support needed later | Compiler abstraction makes this a pluggable backend |
| Font handling differences | Test with the same fonts currently used |
| Typst version breaking changes | Pin compiler version; Typst is post-1.0 roadmap |

## What Improves

- **Hyphenation** — automatic, language-aware (currently none)
- **Microtypography** — character-level spacing adjustment (currently none)
- **Widow/orphan control** — built-in with configurable cost parameters
- **Runt prevention** — avoids single-word last lines
- **PDF quality** — PDF/A support, tagged PDF, proper font subsetting
- **Footnotes** — native support with proper page-bottom placement
- **TOC** — can be added natively
- **Code size** — 7 complex files → 2 simpler files
- **Maintenance** — Typst team maintains the typesetting engine; we just maintain the template
