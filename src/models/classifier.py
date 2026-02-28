"""Citation classification model."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class CitationClassifier(nn.Module):
    """Classify citations as primary or secondary using transformer."""

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Pretrained transformer model name
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability
        """
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)

        Returns:
            Logits for each class
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        return logits

    def predict(self, text: str) -> dict:
        """
        Predict class for a single text input.

        Args:
            text: Input text (citation context)

        Returns:
            Dictionary with predicted class and probabilities
        """
        self.eval()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )

        # Forward pass
        with torch.no_grad():
            logits = self.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                inputs.get('token_type_ids')
            )

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

        class_names = ['secondary', 'primary']

        return {
            'prediction': class_names[pred_class],
            'confidence': probs[0, pred_class].item(),
            'probabilities': {
                'secondary': probs[0, 0].item(),
                'primary': probs[0, 1].item()
            }
        }
