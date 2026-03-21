"""Box model primitives for layout.

Everything on the page is represented as a box. This follows the CSS box model
concept but adapted for print layout. Boxes can contain other boxes, text,
or be atomic (images, rules).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class BoxType(Enum):
    """Types of layout boxes."""
    PAGE = auto()          # A full page
    TEXT_AREA = auto()     # The main text content area of a page
    BLOCK = auto()         # A block-level element (paragraph, heading, etc.)
    LINE = auto()          # A single line of text
    INLINE = auto()        # An inline element within a line
    GLUE = auto()          # Stretchable/shrinkable space (Knuth-Plass term)
    PENALTY = auto()       # Break penalty (Knuth-Plass term)
    IMAGE = auto()         # An image box
    HEADER = auto()        # Running header
    FOOTER = auto()        # Running footer (page numbers, etc.)
    MARGIN_NOTE = auto()   # Margin annotation


@dataclass
class Box:
    """A rectangular region on the page.

    Coordinates are in points, with origin at bottom-left of the page
    (PDF convention). Width and height define the content area.
    """
    box_type: BoxType
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    children: list['Box'] = field(default_factory=list)

    # Content (only one of these should be set)
    text: str = ""
    font_name: str = ""
    font_size: float = 12
    image_path: str = ""

    # Spacing
    margin_top: float = 0
    margin_bottom: float = 0
    padding_top: float = 0
    padding_bottom: float = 0

    # For glue (flexible space)
    stretch: float = 0  # How much this can grow
    shrink: float = 0   # How much this can shrink

    # For penalties
    penalty: float = 0  # Cost of breaking here
    flagged: bool = False  # Is this a hyphenation point?

    # Reference back to document node
    source_node: Optional[object] = None

    def add_child(self, child: 'Box') -> 'Box':
        self.children.append(child)
        return child

    @property
    def total_height(self) -> float:
        """Height including margins and padding."""
        return self.margin_top + self.padding_top + self.height + self.padding_bottom + self.margin_bottom

    @property
    def bottom(self) -> float:
        return self.y - self.height

    def __repr__(self):
        content = f' "{self.text[:20]}"' if self.text else ""
        return f"<{self.box_type.name} {self.width:.0f}x{self.height:.0f}{content}>"


# Knuth-Plass constants for line breaking
INFINITY = 10000

# Standard penalties
PENALTY_MUST_BREAK = -INFINITY    # Force a break here
PENALTY_NO_BREAK = INFINITY       # Never break here
PENALTY_HYPHEN = 50               # Mild discouragement for hyphenation
PENALTY_WIDOW = 150               # Discourage widow lines
PENALTY_ORPHAN = 150              # Discourage orphan lines
PENALTY_CLUB = 150                # Discourage club lines


@dataclass
class GlueSpec:
    """Specification for flexible space (word spacing).

    In the Knuth-Plass model, spaces between words are "glue" that can
    stretch or shrink to achieve justified text.
    """
    width: float = 4.0      # Natural width
    stretch: float = 2.0    # Maximum stretch
    shrink: float = 1.5     # Maximum shrink

    @classmethod
    def inter_word(cls, font_size: float) -> 'GlueSpec':
        """Standard inter-word space for a given font size."""
        space_width = font_size * 0.33
        return cls(
            width=space_width,
            stretch=space_width * 0.5,
            shrink=space_width * 0.33,
        )

    @classmethod
    def inter_sentence(cls, font_size: float) -> 'GlueSpec':
        """Slightly wider space after sentence-ending punctuation."""
        space_width = font_size * 0.40
        return cls(
            width=space_width,
            stretch=space_width * 0.6,
            shrink=space_width * 0.33,
        )


@dataclass
class PageLayout:
    """Describes the layout regions of a single page."""
    page_number: int
    is_recto: bool  # True for right-hand (odd) pages
    content_box: Box = field(default_factory=lambda: Box(BoxType.TEXT_AREA))
    header_box: Optional[Box] = None
    footer_box: Optional[Box] = None
    margin_boxes: list[Box] = field(default_factory=list)

    @property
    def is_verso(self) -> bool:
        return not self.is_recto
