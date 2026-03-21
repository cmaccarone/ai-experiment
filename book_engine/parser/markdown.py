"""Markdown parser that produces a document tree.

Parses Markdown input into our internal document tree representation.
Supports standard Markdown plus book-specific extensions like chapter
markers and front/back matter delimiters.

Book-specific extensions:
  ---frontmatter---    Marks the beginning of front matter
  ---bodymatter---     Marks the beginning of body matter
  ---backmatter---     Marks the beginning of back matter
  # Chapter Title      H1 headings become chapters
  ## Section Title     H2 headings become sections
  [^1]: Footnote text  Footnotes
"""

import re
from typing import Optional

from .document import DocumentNode, NodeType, BookMetadata


def parse_markdown(text: str, metadata: Optional[BookMetadata] = None) -> DocumentNode:
    """Parse a Markdown string into a document tree.

    Args:
        text: The Markdown source text.
        metadata: Optional book metadata (title, author, etc.).

    Returns:
        A DocumentNode of type DOCUMENT containing the full tree.
    """
    doc = DocumentNode(NodeType.DOCUMENT)

    if metadata:
        doc.attributes["metadata"] = metadata

    lines = text.split("\n")
    current_section = doc
    current_chapter = None
    in_code_block = False
    code_block_lines = []
    code_lang = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code block fences
        if line.strip().startswith("```"):
            if in_code_block:
                # End of code block
                code_node = DocumentNode(
                    NodeType.CODE_BLOCK,
                    text="\n".join(code_block_lines),
                    attributes={"language": code_lang},
                )
                current_section.add_child(code_node)
                in_code_block = False
                code_block_lines = []
                code_lang = ""
            else:
                # Start of code block
                in_code_block = True
                code_lang = line.strip()[3:].strip()
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Matter delimiters
        if line.strip() == "---frontmatter---":
            current_section = doc.add_child(DocumentNode(NodeType.FRONT_MATTER))
            i += 1
            continue
        if line.strip() == "---bodymatter---":
            current_section = doc.add_child(DocumentNode(NodeType.BODY_MATTER))
            i += 1
            continue
        if line.strip() == "---backmatter---":
            current_section = doc.add_child(DocumentNode(NodeType.BACK_MATTER))
            i += 1
            continue

        # Page break
        if line.strip() == "---pagebreak---":
            current_section.add_child(DocumentNode(NodeType.PAGE_BREAK))
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            if level == 1:
                # H1 = new chapter
                chapter = DocumentNode(
                    NodeType.CHAPTER,
                    attributes={"title": title},
                )
                heading = DocumentNode(NodeType.HEADING, text=title, attributes={"level": 1})
                chapter.add_child(heading)

                # Add chapter to body matter or directly to doc
                if current_section.node_type == NodeType.BODY_MATTER:
                    current_section.add_child(chapter)
                else:
                    doc.add_child(chapter)

                current_chapter = chapter
                current_section = chapter
            else:
                # H2-H6 = sections within chapter
                section = DocumentNode(
                    NodeType.SECTION,
                    attributes={"title": title, "level": level},
                )
                heading = DocumentNode(NodeType.HEADING, text=title, attributes={"level": level})
                section.add_child(heading)
                current_section.add_child(section)

            i += 1
            continue

        # Block quotes
        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:])
                i += 1
            quote = DocumentNode(NodeType.BLOCK_QUOTE)
            para = DocumentNode(NodeType.PARAGRAPH)
            _parse_inline(para, "\n".join(quote_lines))
            quote.add_child(para)
            current_section.add_child(quote)
            continue

        # Unordered lists
        list_match = re.match(r'^(\s*)[*\-+]\s+(.+)$', line)
        if list_match:
            list_node = DocumentNode(NodeType.LIST, attributes={"ordered": False})
            while i < len(lines) and re.match(r'^(\s*)[*\-+]\s+(.+)$', lines[i]):
                item_match = re.match(r'^(\s*)[*\-+]\s+(.+)$', lines[i])
                item = DocumentNode(NodeType.LIST_ITEM)
                _parse_inline(item, item_match.group(2))
                list_node.add_child(item)
                i += 1
            current_section.add_child(list_node)
            continue

        # Ordered lists
        ol_match = re.match(r'^(\s*)\d+\.\s+(.+)$', line)
        if ol_match:
            list_node = DocumentNode(NodeType.LIST, attributes={"ordered": True})
            while i < len(lines) and re.match(r'^(\s*)\d+\.\s+(.+)$', lines[i]):
                item_match = re.match(r'^(\s*)\d+\.\s+(.+)$', lines[i])
                item = DocumentNode(NodeType.LIST_ITEM)
                _parse_inline(item, item_match.group(2))
                list_node.add_child(item)
                i += 1
            current_section.add_child(list_node)
            continue

        # Horizontal rule (thematic break) — not a matter delimiter
        if re.match(r'^---+\s*$', line) or re.match(r'^\*\*\*+\s*$', line):
            current_section.add_child(DocumentNode(NodeType.PAGE_BREAK))
            i += 1
            continue

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Default: paragraph — collect contiguous non-blank lines
        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
            para_lines.append(lines[i])
            i += 1

        if para_lines:
            para = DocumentNode(NodeType.PARAGRAPH)
            _parse_inline(para, " ".join(para_lines))
            current_section.add_child(para)

    return doc


def _parse_inline(parent: DocumentNode, text: str):
    """Parse inline formatting (bold, italic, code, links) into child nodes."""
    # Simple regex-based inline parser
    pattern = re.compile(
        r'(\*\*(.+?)\*\*)'         # bold
        r'|(\*(.+?)\*)'            # italic
        r'|(`(.+?)`)'              # inline code
        r'|\[([^\]]+)\]\(([^)]+)\)'  # link
        r'|\[\^(\d+)\]'            # footnote ref
    )

    last_end = 0
    for match in pattern.finditer(text):
        # Add any plain text before this match
        if match.start() > last_end:
            parent.add_child(DocumentNode(NodeType.TEXT, text=text[last_end:match.start()]))

        if match.group(2):  # bold
            node = DocumentNode(NodeType.STRONG)
            node.add_child(DocumentNode(NodeType.TEXT, text=match.group(2)))
            parent.add_child(node)
        elif match.group(4):  # italic
            node = DocumentNode(NodeType.EMPHASIS)
            node.add_child(DocumentNode(NodeType.TEXT, text=match.group(4)))
            parent.add_child(node)
        elif match.group(6):  # code
            parent.add_child(DocumentNode(NodeType.CODE, text=match.group(6)))
        elif match.group(7):  # link
            parent.add_child(DocumentNode(
                NodeType.LINK,
                text=match.group(7),
                attributes={"href": match.group(8)},
            ))
        elif match.group(9):  # footnote ref
            parent.add_child(DocumentNode(
                NodeType.FOOTNOTE_REF,
                attributes={"number": int(match.group(9))},
            ))

        last_end = match.end()

    # Trailing plain text
    if last_end < len(text):
        parent.add_child(DocumentNode(NodeType.TEXT, text=text[last_end:]))
