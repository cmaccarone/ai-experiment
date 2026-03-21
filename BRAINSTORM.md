# PDF Book Rendering Engine — Brainstorming

## The Big Question

**What would it take to build a rendering engine that produces truly beautiful, print-ready PDF books from structured input?**

Not "good enough" PDFs. Not "it looks like a Word doc." We're talking about books you'd be proud to put on a shelf — books that feel *typeset*, not generated.

---

## Part 1: Who Is This For?

### Primary Users (in order of priority)

1. **Self-publishing authors** who want print-quality output without learning LaTeX
2. **Small press publishers** who need an automated pipeline
3. **Technical writers** producing manuals, textbooks, documentation
4. **Developers** who want to generate books programmatically (docs-as-code)

### What do they have today?

| User | Current Tools | Pain Points |
|------|---------------|-------------|
| Self-pub author | Word → KDP, Vellum (Mac only), Reedsy | Vellum is Mac-only & expensive; Word output looks amateurish |
| Small press | InDesign, LaTeX | InDesign is manual; LaTeX has steep learning curve |
| Tech writer | Sphinx, mdBook, Docusaurus | These produce web/ePub, not print-quality PDF |
| Developer | Pandoc, WeasyPrint | Output is functional but not beautiful |

### Key insight
There's a gap between "easy but ugly" (Word, Pandoc) and "beautiful but hard" (LaTeX, InDesign). We want to sit right in that gap.

---

## Part 2: What Makes a Book Look "Professionally Typeset"?

This is critical. Most people can't articulate *why* a well-typeset book looks good, but they absolutely notice when it doesn't. Here are the details that matter:

### Typography (the #1 differentiator)

- **Justified text that doesn't look terrible** — This means proper hyphenation and the Knuth-Plass line-breaking algorithm (not greedy). Bad justification = rivers of white space.
- **Proper leading (line spacing)** — Not too tight, not too loose. Should be ~120-145% of font size.
- **Widow and orphan control** — No single lines stranded at top/bottom of pages.
- **Hanging punctuation** — Quotes and hyphens that hang into the margin for optical alignment.
- **Ligatures** — fi, fl, ff, ffi, ffl rendered as proper ligatures.
- **Small caps** — Real small caps, not just scaled-down capitals.
- **Old-style figures** — In body text, numbers should have ascenders/descenders (3, 4, 5 go below baseline).
- **Proper em/en dashes** — With correct spacing.
- **Kerning** — Letter-pair spacing adjustments from the font's kern table.

### Page Layout

- **Consistent text block position** — The text block should be the same size and position on every page. This sounds obvious but many tools get it wrong.
- **Proper margins** — Inner margin (gutter) must be wider for binding. The text block is NOT centered on the page — it's shifted outward.
- **Running headers** — Chapter title on verso, section title on recto (or variations). These should be in small caps or a complementary font.
- **Page numbers** — Positioned consistently. Often bottom-center for front matter (roman numerals) and bottom-outside for body.
- **Baseline grid** — Text on facing pages should align to the same baseline grid, so lines are at the same height across the spread.

### Book Structure

- **Chapters start on recto** — Right-hand page (odd numbered). This means sometimes you get a blank verso.
- **Chapter openers** — The first page of a chapter often has a dropped initial cap, extra whitespace at top, or a decorative element.
- **Front matter order** — Half title → title page → copyright → dedication → TOC → preface
- **Page numbering** — Roman numerals for front matter, arabic for body.
- **No headers on chapter openers** — Running headers are suppressed on the first page of chapters.
- **No headers on blank pages** — Blank verso pages should be truly blank.

### Whitespace & Rhythm

- **Consistent vertical rhythm** — Space between elements should follow a consistent grid (e.g., all multiples of the baseline leading).
- **Section breaks** — Extra space or ornamental breaks between sections, not just a blank line.
- **Paragraph indentation** — First paragraph after a heading is NOT indented. All subsequent paragraphs ARE indented. First line indent, no space between paragraphs.
- **Drop caps** — Optional but classic: first letter of a chapter is enlarged.

---

## Part 3: Input Format

### Option A: Markdown (with extensions)
**Pros:** Everyone knows it, easy tooling, Git-friendly
**Cons:** Limited semantics, would need custom extensions for book features

```markdown
---
title: "My Great Book"
author: "Jane Author"
trim_size: 6x9
---

---frontmatter---

# Dedication

For everyone who...

---bodymatter---

# Chapter One: The Beginning

The story begins here...

## A Section Within

More text...

> A block quote that spans
> multiple lines.

---backmatter---

# Acknowledgments
```

### Option B: Custom DSL
**Pros:** Can be purpose-built for books, rich semantics
**Cons:** Learning curve, tooling from scratch

```
@book(title: "My Great Book", author: "Jane Author")

@frontmatter {
  @dedication { For everyone who... }
}

@chapter(title: "The Beginning") {
  The story begins here...

  @section(title: "A Section Within") {
    More text with @emphasis{inline formatting}.
  }
}
```

### Option C: HTML subset
**Pros:** Powerful, well-understood, CSS for styling
**Cons:** Verbose, not author-friendly

### Option D: Multi-format support
Start with Markdown, add others later. Use an abstract document tree internally.

**Recommendation: Option D (Markdown first, abstract tree internally)**

---

## Part 4: Styling System

### How should users control the look?

1. **Built-in themes** — "Classic", "Modern", "Technical", "Literary" — good defaults that work
2. **YAML/TOML config** — Override individual properties
3. **CSS-like stylesheet** — For power users who want full control
4. **Per-element overrides** — Inline attributes in the input format

```yaml
# book-style.yaml
theme: classic

page:
  trim_size: 6x9
  margins:
    top: 1in
    bottom: 1in
    inside: 1in
    outside: 0.75in
    gutter: 0.25in

typography:
  body:
    font: "Garamond Premier Pro"
    size: 11pt
    leading: 14pt
  headings:
    font: "Futura"
    chapter_size: 24pt

chapters:
  start: recto
  opener_style: drop-cap
  numbering: "Chapter {n}"
```

---

## Part 5: Hard Problems to Solve

### 1. Line Breaking (Knuth-Plass)
The Knuth-Plass algorithm considers ALL possible line breaks in a paragraph simultaneously and finds the globally optimal solution. This is what makes TeX/LaTeX output look so good.

**Complexity:** Medium-High
**Why it matters:** Greedy line breaking (what most tools use) produces visibly worse results — uneven spacing, bad breaks.

### 2. Page Breaking
Similar to line breaking but vertical. Need to consider:
- Widow/orphan control
- Keep-with-next (headings shouldn't be at page bottom)
- Keep-together (don't break a short code block across pages)
- Footnotes (they consume space at page bottom)
- Figures and their captions
- Balance between facing pages

**Complexity:** High
**Why it matters:** Bad page breaks are the most visible sign of automated layout.

### 3. Footnotes
Footnotes are deceptively hard:
- They appear at the bottom of the page where they're referenced
- They consume vertical space from the text area
- A footnote might not fit on the same page as its reference
- Long footnotes might need to split across pages
- The layout engine needs to iteratively adjust: "if I add this footnote, does the text that referenced it still fit on this page?"

**Complexity:** Very High
**Why it matters:** Academic and literary books rely heavily on footnotes.

### 4. Floats (Figures & Tables)
Figures and tables that "float" to optimal positions:
- Top of page, bottom of page, or inline
- Must appear after their first reference
- Must not appear before the paragraph that references them
- Captions must stay with their figure
- Full-page floats

**Complexity:** Very High
**Why it matters:** This is where even LaTeX struggles sometimes.

### 5. Font Handling
- Loading OTF/TTF fonts
- Subsetting fonts for PDF embedding (only include used glyphs)
- Text shaping with HarfBuzz (for complex scripts, ligatures, kerning)
- Fallback fonts for missing glyphs

**Complexity:** High
**Why it matters:** Can't produce professional output with just the 14 built-in PDF fonts.

### 6. Table of Contents
- Generated automatically from headings
- Needs dot leaders (Chapter Title ..... 42)
- Page numbers are only known after layout is complete
- But the TOC is in the front matter, which affects page numbers
- This requires multiple layout passes

**Complexity:** Medium
**Why it matters:** Every book needs one.

### 7. Index Generation
- Authors mark index terms inline
- Index is sorted, formatted, and placed in back matter
- Subentries, cross-references, page ranges
- Like TOC, requires multiple passes

**Complexity:** High
**Why it matters:** Essential for non-fiction/technical books.

---

## Part 6: Architecture Questions

### Rendering approach
1. **Direct PDF generation** (ReportLab, fpdf2, printpdf)
   - Full control, but must implement everything
2. **HTML/CSS → PDF** (WeasyPrint, Prince)
   - Leverage CSS Paged Media, but limited by CSS capabilities
3. **Custom layout + low-level PDF**
   - Build our own layout engine, output raw PDF primitives
   - Most work but most control

**Leaning toward: Option 3** — Custom layout engine with direct PDF output

### Language choice
- **Python** — Rich ecosystem (ReportLab, Pillow, markdown-it), accessible, good for prototyping
- **Rust** — Performance, safety, growing typesetting ecosystem (Typst is Rust)
- **Hybrid** — Python for high-level orchestration, Rust/C for hot paths (text shaping, line breaking)

**Leaning toward: Python first** — faster to prototype, optimize later if needed

### Multi-pass architecture
Books require multiple passes:
1. **Parse pass** — Input → Document Tree
2. **Reference pass** — Resolve cross-references, count figures/tables
3. **Layout pass** — Document Tree → Pages of Boxes
4. **Fixup pass** — Generate TOC, index with now-known page numbers
5. **Re-layout pass** — Re-layout with TOC/index included (may change page numbers)
6. **Render pass** — Pages → PDF

This might need to iterate until page numbers stabilize.

---

## Part 7: MVP Scope

### Phase 1: Proof of Concept
- Markdown input (basic: headings, paragraphs, bold/italic)
- Single font (built-in PDF font)
- Fixed trim size (6x9)
- Basic page breaking (no widow/orphan)
- Page numbers
- Chapter starts on new page
- Output: PDF

### Phase 2: Typographic Quality
- Knuth-Plass line breaking
- Hyphenation
- Widow/orphan control
- Custom fonts (OTF/TTF)
- Proper margins with gutter
- Running headers

### Phase 3: Book Features
- Front matter / back matter
- Table of contents (auto-generated)
- Footnotes
- Block quotes, code blocks, lists
- Drop caps
- Section breaks / ornaments

### Phase 4: Advanced
- Figures and floats
- Tables
- Index generation
- Bibliography
- Multi-column layout
- Themes / style presets
- ePub output (stretch)

---

## Part 8: Open Questions

1. **Should we target a specific print service?** (KDP, IngramSpark, Lulu) — they each have slightly different specs for bleed, margins, etc.

2. **What about ePub?** The document tree could produce both PDF and ePub. Should we plan for this from the start?

3. **Interactive preview?** Should there be a live preview mode (like Typst) where you see changes as you type?

4. **Plugin/extension system?** Allow users to add custom elements, transformations, or output formats?

5. **How much of the LaTeX algorithm do we replicate?** The full Knuth-Plass with all extensions is extremely complex. Where's the 80/20 point?

6. **Accessibility?** Tagged PDF for accessibility (PDF/UA) is increasingly important. Should we include this?

7. **What's our "unfair advantage"?** Why would someone use this over Typst, which already exists and is excellent? Possible answers:
   - Simpler input (Markdown vs Typst syntax)
   - Python ecosystem integration
   - Focus specifically on books (not general documents)
   - Opinionated defaults that produce great output with zero config

---

## Part 9: Competitive Landscape

### Direct competitors
- **Typst** — Modern, fast, excellent output. But general-purpose, not book-focused.
- **Pandoc + LaTeX** — Powerful but requires LaTeX knowledge for customization.
- **Vellum** — Beautiful output, Mac-only, $250+, closed source.
- **Reedsy Book Editor** — Web-based, limited customization.

### Our differentiator could be:
> **"The Vellum experience, but open source, cross-platform, and Markdown-based."**

Beautiful books from Markdown with zero configuration, but infinitely customizable for those who want control.

---

## Next Steps

1. Build a minimal prototype that takes Markdown and produces a basic PDF with proper margins and page breaks
2. Add Knuth-Plass line breaking — this is the single biggest quality differentiator
3. Add hyphenation support
4. Implement chapter openers and running headers
5. Evaluate output quality against commercial tools
