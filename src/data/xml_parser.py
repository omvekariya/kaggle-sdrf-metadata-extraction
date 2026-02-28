"""XML parsing utilities for extracting content from scientific publications."""

from pathlib import Path
from typing import Optional
from lxml import etree
from bs4 import BeautifulSoup


class XMLParser:
    """Parse XML files (JATS, NLM, etc.) and extract content."""

    def __init__(self):
        pass

    def extract_text(self, xml_path: str | Path) -> str:
        """
        Extract all text content from an XML file.

        Args:
            xml_path: Path to the XML file

        Returns:
            Extracted text content
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")

        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml-xml')

        return soup.get_text(separator='\n', strip=True)

    def extract_sections(self, xml_path: str | Path) -> dict[str, str]:
        """
        Extract text organized by sections.

        Args:
            xml_path: Path to the XML file

        Returns:
            Dictionary mapping section titles to content
        """
        xml_path = Path(xml_path)
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml-xml')

        sections = {}

        # Handle JATS/NLM format
        for sec in soup.find_all('sec'):
            title_elem = sec.find('title')
            title = title_elem.get_text(strip=True) if title_elem else 'Untitled'
            content = sec.get_text(separator='\n', strip=True)
            sections[title] = content

        # Extract abstract
        abstract = soup.find('abstract')
        if abstract:
            sections['Abstract'] = abstract.get_text(separator='\n', strip=True)

        return sections

    def extract_references(self, xml_path: str | Path) -> list[dict]:
        """
        Extract reference/citation elements from XML.

        Args:
            xml_path: Path to the XML file

        Returns:
            List of reference dictionaries
        """
        xml_path = Path(xml_path)
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml-xml')

        references = []

        # Find reference section
        ref_list = soup.find('ref-list')
        if ref_list:
            for ref in ref_list.find_all('ref'):
                ref_data = {
                    'id': ref.get('id', ''),
                    'text': ref.get_text(separator=' ', strip=True)
                }

                # Extract DOI if present
                doi = ref.find('pub-id', {'pub-id-type': 'doi'})
                if doi:
                    ref_data['doi'] = doi.get_text(strip=True)

                references.append(ref_data)

        return references

    def extract_data_citations(self, xml_path: str | Path) -> list[dict]:
        """
        Extract data citations and accession numbers.

        Args:
            xml_path: Path to the XML file

        Returns:
            List of data citation dictionaries
        """
        xml_path = Path(xml_path)
        with open(xml_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml-xml')

        citations = []

        # Look for accession numbers
        for acc in soup.find_all('accession'):
            citations.append({
                'type': 'accession',
                'value': acc.get_text(strip=True),
                'database': acc.get('database', 'unknown')
            })

        # Look for data availability statements
        data_avail = soup.find('data-availability')
        if data_avail:
            citations.append({
                'type': 'data_availability',
                'text': data_avail.get_text(separator=' ', strip=True)
            })

        return citations
