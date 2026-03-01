"""
Main pipeline for SDRF metadata extraction from scientific publications.
Kaggle Competition: Harmonizing the Data of your Data
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import Optional
import os


class SDRFExtractor:
    """Extract SDRF metadata from scientific manuscripts."""

    # Metadata columns in submission
    CHARACTERISTICS_COLS = [
        'Characteristics[Age]', 'Characteristics[AlkylationReagent]',
        'Characteristics[AnatomicSiteTumor]', 'Characteristics[AncestryCategory]',
        'Characteristics[BMI]', 'Characteristics[Bait]', 'Characteristics[BiologicalReplicate]',
        'Characteristics[CellLine]', 'Characteristics[CellPart]', 'Characteristics[CellType]',
        'Characteristics[CleavageAgent]', 'Characteristics[Compound]',
        'Characteristics[ConcentrationOfCompound]', 'Characteristics[Depletion]',
        'Characteristics[DevelopmentalStage]', 'Characteristics[DiseaseTreatment]',
        'Characteristics[Disease]', 'Characteristics[GeneticModification]',
        'Characteristics[Genotype]', 'Characteristics[GrowthRate]', 'Characteristics[Label]',
        'Characteristics[MaterialType]', 'Characteristics[Modification]',
        'Characteristics[Modification].1', 'Characteristics[Modification].2',
        'Characteristics[Modification].3', 'Characteristics[Modification].4',
        'Characteristics[Modification].5', 'Characteristics[Modification].6',
        'Characteristics[NumberOfBiologicalReplicates]', 'Characteristics[NumberOfSamples]',
        'Characteristics[NumberOfTechnicalReplicates]', 'Characteristics[OrganismPart]',
        'Characteristics[Organism]', 'Characteristics[OriginSiteDisease]',
        'Characteristics[PooledSample]', 'Characteristics[ReductionReagent]',
        'Characteristics[SamplingTime]', 'Characteristics[Sex]', 'Characteristics[Specimen]',
        'Characteristics[SpikedCompound]', 'Characteristics[Staining]', 'Characteristics[Strain]',
        'Characteristics[SyntheticPeptide]', 'Characteristics[Temperature]',
        'Characteristics[Time]', 'Characteristics[Treatment]', 'Characteristics[TumorCellularity]',
        'Characteristics[TumorGrade]', 'Characteristics[TumorSite]', 'Characteristics[TumorSize]',
        'Characteristics[TumorStage]'
    ]

    COMMENT_COLS = [
        'Comment[AcquisitionMethod]', 'Comment[CollisionEnergy]', 'Comment[EnrichmentMethod]',
        'Comment[FlowRateChromatogram]', 'Comment[FractionIdentifier]',
        'Comment[FractionationMethod]', 'Comment[FragmentMassTolerance]',
        'Comment[FragmentationMethod]', 'Comment[GradientTime]', 'Comment[Instrument]',
        'Comment[IonizationType]', 'Comment[MS2MassAnalyzer]', 'Comment[NumberOfFractions]',
        'Comment[NumberOfMissedCleavages]', 'Comment[PrecursorMassTolerance]', 'Comment[Separation]'
    ]

    FACTORVALUE_COLS = [
        'FactorValue[Bait]', 'FactorValue[CellPart]', 'FactorValue[Compound]',
        'FactorValue[ConcentrationOfCompound].1', 'FactorValue[Disease]',
        'FactorValue[FractionIdentifier]', 'FactorValue[GeneticModification]',
        'FactorValue[Temperature]', 'FactorValue[Treatment]'
    ]

    # Extraction patterns
    PATTERNS = {
        'Characteristics[Organism]': [
            r'(Homo sapiens|human|Mus musculus|mouse|Rattus norvegicus|rat)',
            r'(Plasmodium falciparum|P\. falciparum)',
            r'(Saccharomyces cerevisiae|yeast|S\. cerevisiae)',
            r'(Escherichia coli|E\. coli)',
            r'(Drosophila melanogaster|fruit fly)',
        ],
        'Characteristics[CellLine]': [
            r'(HEK293[T]?|HeLa|U2OS|MCF-?7|A549|K562|Jurkat|THP-?1|SH-SY5Y)',
            r'(ANBL6|MM\.1S|RPMI[ -]?8226|NCI-H929)',
        ],
        'Characteristics[CleavageAgent]': [
            r'(trypsin|Trypsin|Trypsin Gold|chymotrypsin|Lys-?C|Glu-?C|Asp-?N)',
            r'(sequencing[- ]grade modified trypsin)',
        ],
        'Characteristics[AlkylationReagent]': [
            r'(iodoacetamide|IAA|iodoacetic acid|chloroacetamide|CAA)',
            r'(N-ethylmaleimide|NEM)',
        ],
        'Characteristics[ReductionReagent]': [
            r'(DTT|dithiothreitol|TCEP|2-mercaptoethanol|beta-mercaptoethanol)',
        ],
        'Comment[Instrument]': [
            r'(Q[ -]?Exactive|Orbitrap|LTQ|Exploris|Fusion|Eclipse|Astral)',
            r'(Thermo|Bruker|Waters|Sciex|Agilent)',
            r'(timsTOF|TripleTOF|QTOF|maXis)',
        ],
        'Comment[FragmentationMethod]': [
            r'(HCD|CID|ETD|EThcD|UVPD)',
            r'(higher[- ]energy collisional dissociation)',
            r'(collision[- ]induced dissociation)',
            r'(electron transfer dissociation)',
        ],
        'Comment[AcquisitionMethod]': [
            r'(DDA|DIA|PRM|MRM|SRM)',
            r'(data[- ]dependent acquisition)',
            r'(data[- ]independent acquisition)',
        ],
        'Comment[EnrichmentMethod]': [
            r'(IMAC|TiO2|phosphopeptide enrichment|immunoprecipitation|IP)',
            r'(Fe-NTA|Ti-IMAC|phos-tag)',
        ],
        'Characteristics[Modification]': [
            r'(phosphorylation|phospho|Phospho)',
            r'(acetylation|acetyl|Acetyl)',
            r'(ubiquitination|ubiquitin|Ubiquitin)',
            r'(methylation|methyl)',
            r'(Carbamidomethyl|carbamidomethylation)',
            r'(Oxidation|oxidation)',
            r'(TMT|iTRAQ|SILAC)',
        ],
        'Characteristics[OrganismPart]': [
            r'(brain|liver|kidney|heart|lung|spleen|muscle|blood|plasma|serum)',
            r'(cortex|hippocampus|cerebellum)',
        ],
        'Characteristics[Disease]': [
            r'(Alzheimer|cancer|tumor|carcinoma|melanoma|leukemia|myeloma)',
            r'(diabetes|Parkinson|Huntington)',
        ],
        'Comment[PrecursorMassTolerance]': [
            r'(\d+\s*ppm)',
            r'(\d+\.?\d*\s*Da)',
        ],
        'Comment[FragmentMassTolerance]': [
            r'(\d+\.?\d*\s*Da)',
            r'(\d+\s*ppm)',
        ],
        'Comment[GradientTime]': [
            r'(\d+\s*min(?:ute)?s?\s*gradient)',
            r'(gradient.*?(\d+)\s*min)',
        ],
        'Comment[FlowRateChromatogram]': [
            r'(\d+\.?\d*\s*[nμu]L/min)',
            r'(flow rate.*?(\d+\.?\d*)\s*[nμu]L)',
        ],
        'Characteristics[Label]': [
            r'(TMT[- ]?\d*[NC]?|iTRAQ[- ]?\d*)',
            r'(SILAC|label[- ]free)',
            r'(heavy|light|medium)',
        ],
    }

    def __init__(self):
        self.compiled_patterns = {}
        for key, patterns in self.PATTERNS.items():
            self.compiled_patterns[key] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def extract_from_text(self, text: str) -> dict:
        """Extract metadata from manuscript text using regex patterns."""
        extracted = {}

        for col, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(match.group(0))

            if matches:
                # Take the first/most relevant match
                extracted[col] = matches[0]

        return extracted

    def extract_from_filename(self, filename: str) -> dict:
        """Extract metadata hints from raw data filename."""
        extracted = {}

        # Parse filename tokens
        tokens = re.split(r'[_.\-]', filename.replace('.raw', ''))

        # Look for fraction identifiers
        for token in tokens:
            if re.match(r'^[Ff](?:rac(?:tion)?)?(\d+)$', token):
                extracted['Comment[FractionIdentifier]'] = token
            elif re.match(r'^(?:rep|Rep|REP)(\d+)$', token):
                extracted['Characteristics[BiologicalReplicate]'] = token
            elif re.match(r'^(?:TR|tr|tech)(\d+)$', token):
                extracted['Characteristics[TechnicalReplicate]'] = token

        return extracted

    def process_manuscript(self, pub_data: dict) -> dict:
        """Process a single manuscript and extract metadata for all raw files."""
        # Combine relevant text sections
        text_sections = []
        for key in ['TITLE', 'ABSTRACT', 'METHODS']:
            if key in pub_data and pub_data[key]:
                text_sections.append(pub_data[key])

        full_text = '\n'.join(text_sections)

        # Extract global metadata from text
        global_metadata = self.extract_from_text(full_text)

        # Get raw files
        raw_files = pub_data.get('Raw Data Files', [])

        # Create per-file metadata
        results = {}
        for raw_file in raw_files:
            file_metadata = global_metadata.copy()

            # Add filename-specific metadata
            filename_metadata = self.extract_from_filename(raw_file)
            file_metadata.update(filename_metadata)

            results[raw_file] = file_metadata

        return results


def load_test_data(test_dir: Path) -> dict:
    """Load all test manuscripts."""
    test_data = {}

    for json_file in test_dir.glob('PXD*_PubText.json'):
        pxd_id = json_file.stem.replace('_PubText', '')
        with open(json_file, 'r', encoding='utf-8') as f:
            test_data[pxd_id] = json.load(f)

    return test_data


def create_submission(sample_submission: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """Create submission DataFrame from predictions."""
    submission = sample_submission.copy()

    # Drop any unnamed columns (index artifacts)
    submission = submission.loc[:, ~submission.columns.str.contains('^Unnamed')]

    # Get all metadata columns
    metadata_cols = [col for col in submission.columns
                    if col.startswith('Characteristics[') or
                       col.startswith('Comment[') or
                       col.startswith('FactorValue[')]

    # Fill predictions
    for idx, row in submission.iterrows():
        pxd = row['PXD']
        raw_file = row['Raw Data File']

        if pxd in predictions and raw_file in predictions[pxd]:
            file_pred = predictions[pxd][raw_file]

            for col in metadata_cols:
                if col in file_pred:
                    submission.at[idx, col] = file_pred[col]
                else:
                    submission.at[idx, col] = 'Not Applicable'
        else:
            # Default to Not Applicable
            for col in metadata_cols:
                submission.at[idx, col] = 'Not Applicable'

    return submission


def main():
    """Main pipeline execution."""
    # Paths
    data_dir = Path('data/raw')
    test_dir = data_dir / 'Test PubText' / 'Test PubText'
    sample_sub_path = data_dir / 'SampleSubmission.csv'
    output_path = Path('submissions/submission.csv')

    print("Loading test data...")
    test_data = load_test_data(test_dir)
    print(f"Loaded {len(test_data)} test manuscripts")

    print("Loading sample submission...")
    sample_sub = pd.read_csv(sample_sub_path)
    print(f"Sample submission shape: {sample_sub.shape}")

    print("Extracting metadata...")
    extractor = SDRFExtractor()

    predictions = {}
    for pxd_id, pub_data in test_data.items():
        print(f"  Processing {pxd_id}...")
        predictions[pxd_id] = extractor.process_manuscript(pub_data)

    print("Creating submission...")
    submission = create_submission(sample_sub, predictions)

    # Save submission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission


if __name__ == '__main__':
    main()
