"""Style definitions for book elements.

Defines how each document element should be rendered: fonts, sizes, spacing,
alignment, etc. Think of this as CSS for books.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class TextAlign(Enum):
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()
    JUSTIFY = auto()


class VerticalAlign(Enum):
    TOP = auto()
    CENTER = auto()
    BOTTOM = auto()


@dataclass
class FontSpec:
    """Font specification."""
    family: str = "Times"
    size: float = 11          # points
    leading: float = 14       # line height in points
    bold: bool = False
    italic: bool = False
    small_caps: bool = False
    color: str = "#000000"

    @property
    def line_spacing(self) -> float:
        """Space between baselines."""
        return self.leading


@dataclass
class BlockStyle:
    """Style for a block-level element (paragraph, heading, etc.)."""
    font: FontSpec = field(default_factory=FontSpec)
    text_align: TextAlign = TextAlign.JUSTIFY
    space_before: float = 0     # points before the block
    space_after: float = 6      # points after the block
    first_line_indent: float = 18  # points, typical paragraph indent
    left_indent: float = 0
    right_indent: float = 0
    keep_with_next: bool = False   # Don't break page between this and next
    keep_together: bool = False    # Don't break this block across pages
    page_break_before: bool = False


@dataclass
class BookStylesheet:
    """Complete stylesheet for a book.

    Maps document element types to their styles.
    """
    # Body text
    body: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Times", size=11, leading=14),
        text_align=TextAlign.JUSTIFY,
        first_line_indent=18,
        space_after=0,
    ))

    # First paragraph after heading (no indent, traditionally)
    body_first: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Times", size=11, leading=14),
        text_align=TextAlign.JUSTIFY,
        first_line_indent=0,
        space_after=0,
    ))

    # Chapter title
    chapter_title: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=24, leading=28, bold=True),
        text_align=TextAlign.LEFT,
        space_before=144,   # 2 inches from top
        space_after=36,
        first_line_indent=0,
        keep_with_next=True,
    ))

    # Section headings (h2)
    section_heading: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=16, leading=20, bold=True),
        text_align=TextAlign.LEFT,
        space_before=24,
        space_after=12,
        first_line_indent=0,
        keep_with_next=True,
    ))

    # Subsection headings (h3)
    subsection_heading: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=13, leading=16, bold=True),
        text_align=TextAlign.LEFT,
        space_before=18,
        space_after=8,
        first_line_indent=0,
        keep_with_next=True,
    ))

    # Block quotes
    block_quote: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Times", size=10, leading=13, italic=True),
        text_align=TextAlign.JUSTIFY,
        left_indent=36,
        right_indent=36,
        space_before=12,
        space_after=12,
        first_line_indent=0,
    ))

    # Code blocks
    code_block: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Courier", size=9, leading=11),
        text_align=TextAlign.LEFT,
        left_indent=18,
        space_before=12,
        space_after=12,
        first_line_indent=0,
        keep_together=True,
    ))

    # Footnotes
    footnote: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Times", size=8, leading=10),
        text_align=TextAlign.JUSTIFY,
        first_line_indent=0,
        space_after=2,
    ))

    # Running headers
    running_header: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=8, leading=10, small_caps=True),
        text_align=TextAlign.CENTER,
        first_line_indent=0,
    ))

    # Page numbers
    page_number: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Times", size=10, leading=12),
        text_align=TextAlign.CENTER,
        first_line_indent=0,
    ))

    # Title page
    title: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=36, leading=40, bold=True),
        text_align=TextAlign.CENTER,
        space_before=216,  # 3 inches from top
        space_after=18,
        first_line_indent=0,
    ))

    subtitle: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=18, leading=22),
        text_align=TextAlign.CENTER,
        space_after=36,
        first_line_indent=0,
    ))

    author: BlockStyle = field(default_factory=lambda: BlockStyle(
        font=FontSpec(family="Helvetica", size=16, leading=20),
        text_align=TextAlign.CENTER,
        space_before=72,
        first_line_indent=0,
    ))
