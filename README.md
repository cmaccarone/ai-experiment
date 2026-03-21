# PDF Book Rendering Engine

An exploration into building a print-ready PDF book rendering engine.

## Vision

A engine that takes structured content (Markdown, HTML, or a custom DSL) and produces beautifully typeset, print-ready PDF books — handling all the complexities that make book layout hard.

## Core Challenges

### 1. Page Layout & Pagination
- Fixed page dimensions (trim size: 6x9", 5.5x8.5", etc.)
- Margins with gutter (inner margin wider for binding)
- Headers and footers (running heads, page numbers)
- Recto/verso (right/left) page awareness

### 2. Typography
- Font selection and embedding
- Hyphenation and justification (H&J)
- Kerning, ligatures, and OpenType features
- Widow and orphan control
- Optical margin alignment (hanging punctuation)

### 3. Content Flow
- Paragraphs that break across pages
- Footnotes that float to page bottom
- Figures/images with captions and placement rules
- Tables that may span pages
- Block quotes, code blocks, lists

### 4. Book Structure
- Front matter (title page, copyright, TOC, dedication)
- Chapters with configurable openers (recto-start, any-start)
- Part divisions
- Back matter (index, glossary, bibliography)
- Automatic table of contents generation

### 5. Cross-References & Links
- Internal references (figures, tables, chapters)
- Footnote/endnote numbering
- Index generation
- Bibliography/citation management

## Architecture

```
Input (Markdown/HTML/DSL)
        |
        v
   +---------+
   |  Parser  |  -> Abstract Document Tree
   +----+-----+
        |
        v
   +--------------+
   | Style Engine  |  -> Resolve fonts, spacing, colors
   +------+-------+
        |
        v
   +--------------+
   | Layout Engine |  -> Box model, line breaking, pagination
   +------+-------+
        |
        v
   +--------------+
   | PDF Renderer  |  -> Generate PDF output
   +--------------+
```

## Prior Art & Inspiration

| Tool | Approach | Strengths | Weaknesses |
|------|----------|-----------|------------|
| LaTeX | TeX typesetting | Gold standard typography | Arcane syntax, hard to customize |
| WeasyPrint | CSS + HTML to PDF | Web standards based | Limited book-specific features |
| Paged.js | CSS Paged Media in browser | Modern CSS approach | Browser rendering limitations |
| Prince XML | CSS to PDF | Excellent CSS support | Commercial, closed source |
| Typst | Modern typesetting | Clean syntax, fast | Younger ecosystem |
| SILE | Lua-based typesetting | Flexible, TeX-quality | Smaller community |

## Getting Started

```bash
pip install -r requirements.txt
python -m book_engine.cli sample.md -o output.pdf
```

## Project Structure

```
book_engine/
  __init__.py
  cli.py              # Command-line interface
  parser/
    __init__.py
    markdown.py        # Markdown -> Document Tree
    document.py        # Document tree data structures
  style/
    __init__.py
    stylesheet.py      # Style definitions
    resolver.py        # Style cascade & resolution
  layout/
    __init__.py
    engine.py          # Main layout algorithm
    line_breaking.py   # Knuth-Plass line breaking
    pagination.py      # Page breaking logic
    boxes.py           # Box model primitives
  renderer/
    __init__.py
    pdf.py             # PDF output generation
  fonts/
    __init__.py
    manager.py         # Font loading & subsetting
```
