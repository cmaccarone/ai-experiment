"""Document tree data structures.

The document tree is the intermediate representation between parsed input
and layout. It represents the logical structure of the book without any
information about how it will be rendered on pages.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class NodeType(Enum):
    """Types of nodes in the document tree."""
    DOCUMENT = auto()
    FRONT_MATTER = auto()
    BODY_MATTER = auto()
    BACK_MATTER = auto()
    PART = auto()
    CHAPTER = auto()
    SECTION = auto()
    PARAGRAPH = auto()
    HEADING = auto()
    BLOCK_QUOTE = auto()
    CODE_BLOCK = auto()
    LIST = auto()
    LIST_ITEM = auto()
    TABLE = auto()
    TABLE_ROW = auto()
    TABLE_CELL = auto()
    FIGURE = auto()
    IMAGE = auto()
    FOOTNOTE = auto()
    TITLE_PAGE = auto()
    TOC = auto()
    # Inline nodes
    TEXT = auto()
    EMPHASIS = auto()
    STRONG = auto()
    CODE = auto()
    LINK = auto()
    FOOTNOTE_REF = auto()
    PAGE_BREAK = auto()


class ChapterStart(Enum):
    """Where a chapter can start."""
    ANY = auto()       # Next available page
    RECTO = auto()     # Right-hand (odd) page only
    VERSO = auto()     # Left-hand (even) page only


@dataclass
class DocumentNode:
    """A node in the document tree.

    This is the universal building block. Each node has a type, optional text
    content, optional attributes, and zero or more children.
    """
    node_type: NodeType
    children: list['DocumentNode'] = field(default_factory=list)
    text: str = ""
    attributes: dict = field(default_factory=dict)
    parent: Optional['DocumentNode'] = None

    def add_child(self, child: 'DocumentNode') -> 'DocumentNode':
        child.parent = self
        self.children.append(child)
        return child

    def walk(self):
        """Depth-first traversal of this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find_all(self, node_type: NodeType):
        """Find all descendant nodes of a given type."""
        return [node for node in self.walk() if node.node_type == node_type]

    def __repr__(self):
        text_preview = f' "{self.text[:30]}..."' if len(self.text) > 30 else f' "{self.text}"' if self.text else ""
        return f"<{self.node_type.name}{text_preview} ({len(self.children)} children)>"


@dataclass
class BookMetadata:
    """Metadata for the entire book."""
    title: str = ""
    subtitle: str = ""
    author: str = ""
    publisher: str = ""
    isbn: str = ""
    copyright_year: str = ""
    language: str = "en"
    dedication: str = ""


@dataclass
class TrimSize:
    """Physical dimensions of the book page in points (1 point = 1/72 inch)."""
    width: float
    height: float

    @classmethod
    def from_inches(cls, width: float, height: float) -> 'TrimSize':
        return cls(width=width * 72, height=height * 72)

    # Common book sizes
    @classmethod
    def trade_paperback(cls) -> 'TrimSize':
        """6 x 9 inches — most common trade paperback."""
        return cls.from_inches(6, 9)

    @classmethod
    def digest(cls) -> 'TrimSize':
        """5.5 x 8.5 inches — digest size."""
        return cls.from_inches(5.5, 8.5)

    @classmethod
    def mass_market(cls) -> 'TrimSize':
        """4.25 x 6.87 inches — mass market paperback."""
        return cls.from_inches(4.25, 6.87)

    @classmethod
    def us_letter(cls) -> 'TrimSize':
        """8.5 x 11 inches — US Letter."""
        return cls.from_inches(8.5, 11)

    @classmethod
    def a5(cls) -> 'TrimSize':
        """148 x 210 mm — ISO A5."""
        return cls(width=419.53, height=595.28)


@dataclass
class Margins:
    """Page margins in points. Gutter is extra inner margin for binding."""
    top: float = 72       # 1 inch
    bottom: float = 72
    inside: float = 72    # Toward the spine
    outside: float = 54   # Away from the spine
    gutter: float = 18    # Extra binding margin

    @property
    def left_on_recto(self) -> float:
        """Left margin on a right-hand (odd) page."""
        return self.inside + self.gutter

    @property
    def right_on_recto(self) -> float:
        return self.outside

    @property
    def left_on_verso(self) -> float:
        """Left margin on a left-hand (even) page."""
        return self.outside

    @property
    def right_on_verso(self) -> float:
        return self.inside + self.gutter


@dataclass
class BookSpec:
    """Complete specification for a book's physical properties."""
    trim_size: TrimSize = field(default_factory=TrimSize.trade_paperback)
    margins: Margins = field(default_factory=Margins)
    metadata: BookMetadata = field(default_factory=BookMetadata)
    chapter_start: ChapterStart = ChapterStart.RECTO

    @property
    def text_width(self) -> float:
        """Available width for text on a recto page."""
        return self.trim_size.width - self.margins.left_on_recto - self.margins.right_on_recto

    @property
    def text_height(self) -> float:
        """Available height for text content."""
        return self.trim_size.height - self.margins.top - self.margins.bottom
