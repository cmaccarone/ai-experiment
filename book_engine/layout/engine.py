"""Main layout engine — transforms document tree + styles into positioned boxes.

This is the heart of the rendering engine. It takes the logical document tree
and the stylesheet, and produces a sequence of pages filled with positioned
boxes ready for PDF rendering.

The layout process:
1. Walk the document tree
2. For each node, resolve its style
3. Shape text into lines (line breaking)
4. Fill pages with lines and blocks (page breaking)
5. Position headers, footers, page numbers
"""

from ..parser.document import DocumentNode, NodeType, BookSpec, ChapterStart
from ..style.stylesheet import BookStylesheet, BlockStyle, TextAlign
from .boxes import Box, BoxType, PageLayout


class LayoutEngine:
    """Lays out a document tree into pages of positioned boxes."""

    def __init__(self, spec: BookSpec, stylesheet: BookStylesheet):
        self.spec = spec
        self.stylesheet = stylesheet
        self.pages: list[PageLayout] = []
        self.current_page: PageLayout | None = None
        self.cursor_y: float = 0  # Current vertical position (top-down)

    def layout(self, document: DocumentNode) -> list[PageLayout]:
        """Layout the entire document, returning a list of pages."""
        self.pages = []
        self._new_page()

        for node in document.children:
            self._layout_node(node)

        return self.pages

    def _new_page(self, force_recto: bool = False):
        """Start a new page."""
        page_num = len(self.pages) + 1

        # If we need a recto page and this would be verso, add a blank
        if force_recto and page_num % 2 == 0:
            blank = PageLayout(
                page_number=page_num,
                is_recto=False,
            )
            self.pages.append(blank)
            page_num += 1

        is_recto = (page_num % 2 == 1)

        # Determine margins based on page side
        if is_recto:
            left = self.spec.margins.left_on_recto
        else:
            left = self.spec.margins.left_on_verso

        content_box = Box(
            box_type=BoxType.TEXT_AREA,
            x=left,
            y=self.spec.trim_size.height - self.spec.margins.top,
            width=self.spec.text_width,
            height=self.spec.text_height,
        )

        page = PageLayout(
            page_number=page_num,
            is_recto=is_recto,
            content_box=content_box,
        )
        self.pages.append(page)
        self.current_page = page
        self.cursor_y = 0  # Distance from top of content area

    def _remaining_height(self) -> float:
        """How much vertical space is left on the current page."""
        return self.spec.text_height - self.cursor_y

    def _layout_node(self, node: DocumentNode):
        """Dispatch layout for a single node."""
        match node.node_type:
            case NodeType.CHAPTER:
                self._layout_chapter(node)
            case NodeType.SECTION:
                self._layout_section(node)
            case NodeType.PARAGRAPH:
                self._layout_paragraph(node, self.stylesheet.body)
            case NodeType.HEADING:
                self._layout_heading(node)
            case NodeType.BLOCK_QUOTE:
                self._layout_block_quote(node)
            case NodeType.CODE_BLOCK:
                self._layout_code_block(node)
            case NodeType.LIST:
                self._layout_list(node)
            case NodeType.PAGE_BREAK:
                self._new_page()
            case NodeType.FRONT_MATTER | NodeType.BODY_MATTER | NodeType.BACK_MATTER:
                for child in node.children:
                    self._layout_node(child)
            case _:
                # For unhandled types, recurse into children
                for child in node.children:
                    self._layout_node(child)

    def _layout_chapter(self, node: DocumentNode):
        """Layout a chapter — starts on a new page (recto by default)."""
        force_recto = self.spec.chapter_start == ChapterStart.RECTO
        self._new_page(force_recto=force_recto)

        for child in node.children:
            self._layout_node(child)

    def _layout_heading(self, node: DocumentNode):
        """Layout a heading."""
        level = node.attributes.get("level", 1)
        if level == 1:
            style = self.stylesheet.chapter_title
        elif level == 2:
            style = self.stylesheet.section_heading
        else:
            style = self.stylesheet.subsection_heading

        self._layout_paragraph(node, style)

    def _layout_section(self, node: DocumentNode):
        """Layout a section and its children."""
        for child in node.children:
            self._layout_node(child)

    def _layout_paragraph(self, node: DocumentNode, style: BlockStyle):
        """Layout a paragraph as a block box containing line boxes.

        This is a simplified version — a real implementation would use
        Knuth-Plass line breaking. Here we do a greedy line break.
        """
        # Collect all text from inline children
        text = self._extract_text(node)
        if not text.strip():
            return

        # Add space before
        self.cursor_y += style.space_before

        # Check if we need a new page
        min_height = style.font.leading * 2  # At least 2 lines
        if self._remaining_height() < min_height:
            self._new_page()

        # Simple greedy line breaking
        available_width = self.spec.text_width - style.left_indent - style.right_indent
        words = text.split()
        lines = []
        current_line: list[str] = []
        current_width = style.first_line_indent if style.first_line_indent else 0

        # Approximate character width (proper implementation would use font metrics)
        char_width = style.font.size * 0.5

        for word in words:
            word_width = len(word) * char_width
            space_width = char_width

            if current_line and (current_width + space_width + word_width > available_width):
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                if current_line:
                    current_width += space_width
                current_line.append(word)
                current_width += word_width

        if current_line:
            lines.append(" ".join(current_line))

        # Create line boxes
        content_x = self.current_page.content_box.x + style.left_indent
        for i, line_text in enumerate(lines):
            # Check for page break
            if self._remaining_height() < style.font.leading:
                self._new_page()

            line_y = self.current_page.content_box.y - self.cursor_y

            indent = style.first_line_indent if i == 0 else 0

            line_box = Box(
                box_type=BoxType.LINE,
                x=content_x + indent,
                y=line_y,
                width=available_width - indent,
                height=style.font.leading,
                text=line_text,
                font_name=self._resolve_font_name(style),
                font_size=style.font.size,
                source_node=node,
            )
            self.current_page.content_box.add_child(line_box)
            self.cursor_y += style.font.leading

        # Add space after
        self.cursor_y += style.space_after

    def _layout_block_quote(self, node: DocumentNode):
        for child in node.children:
            if child.node_type == NodeType.PARAGRAPH:
                self._layout_paragraph(child, self.stylesheet.block_quote)

    def _layout_code_block(self, node: DocumentNode):
        self._layout_paragraph(node, self.stylesheet.code_block)

    def _layout_list(self, node: DocumentNode):
        ordered = node.attributes.get("ordered", False)
        for idx, item in enumerate(node.children):
            if item.node_type == NodeType.LIST_ITEM:
                prefix = f"{idx + 1}. " if ordered else "- "
                # Temporarily prepend bullet/number
                text = prefix + self._extract_text(item)
                temp_node = DocumentNode(NodeType.PARAGRAPH, text=text)
                temp_node.add_child(DocumentNode(NodeType.TEXT, text=text))
                style = BlockStyle(
                    font=self.stylesheet.body.font,
                    text_align=TextAlign.LEFT,
                    first_line_indent=0,
                    left_indent=18,
                    space_after=2,
                )
                self._layout_paragraph(temp_node, style)

    def _extract_text(self, node: DocumentNode) -> str:
        """Recursively extract plain text from a node."""
        if node.text:
            return node.text
        parts = []
        for child in node.children:
            parts.append(self._extract_text(child))
        return " ".join(parts).strip()

    def _resolve_font_name(self, style: BlockStyle) -> str:
        """Resolve a font spec to a PDF font name."""
        family = style.font.family
        if style.font.bold and style.font.italic:
            return f"{family}-BoldOblique"
        elif style.font.bold:
            return f"{family}-Bold"
        elif style.font.italic:
            return f"{family}-Oblique"
        return family
