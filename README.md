# Harmonizing the Data of your Data - Kaggle Competition

## Competition Overview

This competition is part of the **Make Data Count (MDC)** initiative, aiming to identify and contextualize data citations in scientific publications.

### Task
- Process scientific articles (PDF or XML format)
- Extract data citations (DOIs, accession numbers like PDB, GenBank, Dryad, etc.)
- Classify citations as:
  - **Primary**: Data published/generated in the paper
  - **Secondary**: Reuse of existing data

### Evaluation
- Submissions must be open-source LLM solutions
- Run against hidden test set of PDFs/XMLs

## Project Structure

```
├── data/
│   ├── raw/              # Original competition data
│   ├── processed/        # Cleaned/transformed data
│   └── external/         # External datasets
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model architectures
│   ├── features/         # Feature engineering
│   └── utils/            # Utility functions
├── models/               # Saved model weights
├── submissions/          # Submission files
├── configs/              # Configuration files
└── requirements.txt      # Python dependencies
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download competition data
kaggle competitions download -c harmonizing-the-data-of-your-data
unzip harmonizing-the-data-of-your-data.zip -d data/raw/
```

## Links

- [Competition Page](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)
- [Make Data Count](https://makedatacount.org/)
