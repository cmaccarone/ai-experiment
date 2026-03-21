"""PDF renderer — takes laid-out pages and produces a PDF file.

Uses ReportLab to generate the actual PDF output from the positioned
box tree produced by the layout engine.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

from ..layout.boxes import Box, BoxType, PageLayout
from ..parser.document import BookSpec


class PDFRenderer:
    """Renders laid-out pages to a PDF file."""

    def __init__(self, spec: BookSpec):
        self.spec = spec

    def render(self, pages: list[PageLayout], output_path: str):
        """Render all pages to a PDF file."""
        page_size = (self.spec.trim_size.width, self.spec.trim_size.height)
        c = canvas.Canvas(output_path, pagesize=page_size)

        c.setTitle(self.spec.metadata.title or "Untitled")
        c.setAuthor(self.spec.metadata.author or "")

        for page in pages:
            self._render_page(c, page)
            c.showPage()

        c.save()

    def _render_page(self, c: canvas.Canvas, page: PageLayout):
        """Render a single page."""
        # Render page number
        self._render_page_number(c, page)

        # Render all content boxes
        self._render_box(c, page.content_box)

    def _render_box(self, c: canvas.Canvas, box: Box):
        """Recursively render a box and its children."""
        match box.box_type:
            case BoxType.LINE:
                self._render_line(c, box)
            case BoxType.IMAGE:
                self._render_image(c, box)
            case _:
                # Container box — just render children
                for child in box.children:
                    self._render_box(c, child)

    def _render_line(self, c: canvas.Canvas, box: Box):
        """Render a line of text."""
        if not box.text:
            return

        font_name = box.font_name or "Times-Roman"
        # Map our font names to ReportLab built-in names
        font_map = {
            "Times": "Times-Roman",
            "Times-Bold": "Times-Bold",
            "Times-Oblique": "Times-Italic",
            "Times-BoldOblique": "Times-BoldItalic",
            "Helvetica": "Helvetica",
            "Helvetica-Bold": "Helvetica-Bold",
            "Helvetica-Oblique": "Helvetica-Oblique",
            "Helvetica-BoldOblique": "Helvetica-BoldOblique",
            "Courier": "Courier",
            "Courier-Bold": "Courier-Bold",
            "Courier-Oblique": "Courier-Oblique",
        }
        pdf_font = font_map.get(font_name, "Times-Roman")

        c.setFont(pdf_font, box.font_size)
        # box.y is the top of the line; text baseline needs to be offset
        baseline_y = box.y - box.font_size
        c.drawString(box.x, baseline_y, box.text)

    def _render_image(self, c: canvas.Canvas, box: Box):
        """Render an image box."""
        if box.image_path:
            c.drawImage(
                box.image_path,
                box.x, box.y - box.height,
                width=box.width,
                height=box.height,
            )

    def _render_page_number(self, c: canvas.Canvas, page: PageLayout):
        """Render the page number in the footer."""
        if page.page_number <= 1:
            return  # Skip page number on first page

        c.setFont("Times-Roman", 10)
        page_str = str(page.page_number)

        center_x = self.spec.trim_size.width / 2
        y = self.spec.margins.bottom / 2

        if page.is_recto:
            # Right-align on recto pages
            x = self.spec.trim_size.width - self.spec.margins.outside
            c.drawRightString(x, y, page_str)
        else:
            # Left-align on verso pages
            x = self.spec.margins.outside
            c.drawString(x, y, page_str)
