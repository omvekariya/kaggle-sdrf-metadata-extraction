"""Citation extraction model using LLM."""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """Represents an extracted data citation."""
    identifier: str           # DOI or accession number
    identifier_type: str      # 'doi', 'pdb', 'genbank', 'dryad', etc.
    classification: str       # 'primary' or 'secondary'
    confidence: float         # Model confidence score
    context: str              # Surrounding text context


class CitationExtractor:
    """Extract data citations from scientific text using LLM."""

    # Common patterns for data identifiers
    PATTERNS = {
        'doi': r'10\.\d{4,}/[^\s\]>]+',
        'pdb': r'\b[0-9][A-Za-z0-9]{3}\b',  # PDB accession (e.g., 6TAP)
        'genbank': r'\b[A-Z]{1,2}\d{5,8}\b',
        'sra': r'\b[SED]R[RPXS]\d{6,}\b',  # SRA accession
        'geo': r'\bGSE\d+\b',  # GEO accession
        'arrayexpress': r'\bE-[A-Z]{4}-\d+\b',
        'dryad': r'dryad\.[a-z0-9]+',
        'zenodo': r'zenodo\.\d+',
        'figshare': r'figshare\.\d+',
    }

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the extractor.

        Args:
            model_name: Name of the LLM to use
        """
        self.model_name = model_name
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def extract_identifiers(self, text: str) -> list[dict]:
        """
        Extract potential data identifiers using regex patterns.

        Args:
            text: Input text to search

        Returns:
            List of found identifiers with their types
        """
        identifiers = []

        for id_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                identifiers.append({
                    'identifier': match.group(),
                    'type': id_type,
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-100):match.end()+100]
                })

        return identifiers

    def classify_citation(self, citation: dict, full_text: str) -> str:
        """
        Classify a citation as primary or secondary.

        Args:
            citation: Citation dictionary with identifier and context
            full_text: Full document text for context

        Returns:
            'primary' or 'secondary'
        """
        # TODO: Implement LLM-based classification
        # For now, use heuristics
        context_lower = citation.get('context', '').lower()

        primary_indicators = [
            'we deposited', 'we submitted', 'data are available',
            'our data', 'this study', 'we generated',
            'deposited in', 'submitted to', 'accession number'
        ]

        secondary_indicators = [
            'previously published', 'downloaded from',
            'obtained from', 'retrieved from', 'from the',
            'publicly available', 'prior study', 'previous work'
        ]

        primary_score = sum(1 for ind in primary_indicators if ind in context_lower)
        secondary_score = sum(1 for ind in secondary_indicators if ind in context_lower)

        if primary_score > secondary_score:
            return 'primary'
        elif secondary_score > primary_score:
            return 'secondary'
        else:
            return 'unknown'

    def extract_and_classify(self, text: str) -> list[Citation]:
        """
        Extract all citations and classify them.

        Args:
            text: Full document text

        Returns:
            List of Citation objects
        """
        identifiers = self.extract_identifiers(text)
        citations = []

        for ident in identifiers:
            classification = self.classify_citation(ident, text)
            citations.append(Citation(
                identifier=ident['identifier'],
                identifier_type=ident['type'],
                classification=classification,
                confidence=0.5,  # TODO: Get from model
                context=ident['context']
            ))

        return citations
