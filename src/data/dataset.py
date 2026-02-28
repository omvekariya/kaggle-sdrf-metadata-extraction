"""Dataset classes for training and inference."""

import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset


class CitationDataset(Dataset):
    """Dataset for data citation extraction and classification."""

    def __init__(
        self,
        data_dir: str | Path,
        labels_file: Optional[str | Path] = None,
        transform=None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing PDF/XML files
            labels_file: Path to labels CSV (for training)
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Collect all PDF and XML files
        self.pdf_files = list(self.data_dir.glob("**/*.pdf"))
        self.xml_files = list(self.data_dir.glob("**/*.xml"))
        self.all_files = self.pdf_files + self.xml_files

        # Load labels if provided
        self.labels = None
        if labels_file:
            self.labels = pd.read_csv(labels_file)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.all_files[idx]

        item = {
            'file_path': str(file_path),
            'file_type': file_path.suffix.lower(),
            'file_name': file_path.name
        }

        if self.labels is not None:
            # Match labels by filename
            file_labels = self.labels[
                self.labels['filename'] == file_path.name
            ]
            if not file_labels.empty:
                item['citations'] = file_labels.to_dict('records')

        if self.transform:
            item = self.transform(item)

        return item

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            'total_files': len(self.all_files),
            'pdf_files': len(self.pdf_files),
            'xml_files': len(self.xml_files),
            'has_labels': self.labels is not None,
            'num_labels': len(self.labels) if self.labels is not None else 0
        }
