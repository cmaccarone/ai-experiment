"""Command-line interface for the book rendering engine."""

import argparse
import sys

from .parser.document import BookSpec, BookMetadata, TrimSize
from .parser.markdown import parse_markdown
from .style.stylesheet import BookStylesheet
from .layout.engine import LayoutEngine
from .renderer.pdf import PDFRenderer


def main():
    parser = argparse.ArgumentParser(
        description="Render Markdown to a print-ready PDF book.",
    )
    parser.add_argument("input", help="Input Markdown file")
    parser.add_argument("-o", "--output", default="output.pdf", help="Output PDF file")
    parser.add_argument("--title", default="", help="Book title")
    parser.add_argument("--author", default="", help="Book author")
    parser.add_argument(
        "--size",
        choices=["trade", "digest", "mass-market", "letter", "a5"],
        default="trade",
        help="Trim size (default: trade = 6x9 inches)",
    )
    args = parser.parse_args()

    # Read input
    try:
        with open(args.input, "r") as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Configure book
    trim_sizes = {
        "trade": TrimSize.trade_paperback,
        "digest": TrimSize.digest,
        "mass-market": TrimSize.mass_market,
        "letter": TrimSize.us_letter,
        "a5": TrimSize.a5,
    }

    metadata = BookMetadata(title=args.title, author=args.author)
    spec = BookSpec(
        trim_size=trim_sizes[args.size](),
        metadata=metadata,
    )
    stylesheet = BookStylesheet()

    # Parse
    print(f"Parsing {args.input}...")
    doc = parse_markdown(markdown_text, metadata)

    # Layout
    print("Laying out pages...")
    engine = LayoutEngine(spec, stylesheet)
    pages = engine.layout(doc)
    print(f"  -> {len(pages)} pages")

    # Render
    print(f"Rendering to {args.output}...")
    renderer = PDFRenderer(spec)
    renderer.render(pages, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
