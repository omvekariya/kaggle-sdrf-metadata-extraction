"""PDF parsing utilities for extracting text from scientific publications."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional


class PDFParser:
    """Parse PDF files and extract text content."""

    def __init__(self):
        pass

    def extract_text(self, pdf_path: str | Path) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())

        return "\n".join(text_parts)

    def extract_text_by_page(self, pdf_path: str | Path) -> list[str]:
        """
        Extract text from each page separately.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of text content per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text())

        return pages

    def get_metadata(self, pdf_path: str | Path) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary of PDF metadata
        """
        pdf_path = Path(pdf_path)
        with fitz.open(pdf_path) as doc:
            return doc.metadata
