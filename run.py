#!/usr/bin/env python3
"""Main entry point for the competition pipeline."""

import argparse
from pathlib import Path

from src.data.pdf_parser import PDFParser
from src.data.xml_parser import XMLParser
from src.data.dataset import CitationDataset
from src.models.citation_extractor import CitationExtractor
from src.utils.helpers import setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Harmonizing the Data of your Data - Citation Extraction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/submission.csv",
        help="Output file path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extract", "train", "predict"],
        default="extract",
        help="Operation mode"
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    config = load_config(args.config)
    logger.info(f"Running in {args.mode} mode")

    input_path = Path(args.input)

    if args.mode == "extract":
        # Initialize extractors
        pdf_parser = PDFParser()
        xml_parser = XMLParser()
        extractor = CitationExtractor()

        results = []

        # Process files
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob("**/*.pdf")) + list(input_path.glob("**/*.xml"))

        for file_path in files:
            logger.info(f"Processing: {file_path.name}")

            # Extract text
            if file_path.suffix.lower() == ".pdf":
                text = pdf_parser.extract_text(file_path)
            else:
                text = xml_parser.extract_text(file_path)

            # Extract and classify citations
            citations = extractor.extract_and_classify(text)

            for citation in citations:
                results.append({
                    "filename": file_path.name,
                    "identifier": citation.identifier,
                    "identifier_type": citation.identifier_type,
                    "classification": citation.classification,
                    "confidence": citation.confidence
                })

        # Save results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")

    elif args.mode == "train":
        logger.info("Training mode not yet implemented")
        # TODO: Implement training pipeline

    elif args.mode == "predict":
        logger.info("Predict mode not yet implemented")
        # TODO: Implement prediction pipeline


if __name__ == "__main__":
    main()
