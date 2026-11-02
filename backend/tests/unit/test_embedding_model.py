import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.app.embeddings.model import EmbeddingModel

@pytest.fixture
def mock_model():
    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 384

    fake_model.encode.side_effect = lambda texts, **kwargs: (
        np.ones((len(texts), 384), dtype=np.float32)
        if isinstance(texts, list)
        else np.ones(384, dtype=np.float32)
    )

    with patch(
        "backend.app.embeddings.model.SentenceTransformer",
        return_value=fake_model,
    ):
        yield fake_model

def test_model_initialization(mock_model):
    model = EmbeddingModel()

    assert model.embedding_dimension == 384
    assert model.model_name == EmbeddingModel.DEFAULT_MODEL_NAME

def test_encode_multiple_tests(mock_model):
    model = EmbeddingModel()

    embeddings = model.encode(["First document", "Second document"])

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32

def test_encode_query(mock_model):
    model = EmbeddingModel()

    embeddings = model.encode_query("What is the placement process?")

    assert embeddings.shape == (384,)
    assert embeddings.dtype == np.float32

def test_encode_rejects_empty_list(mock_model):
    model = EmbeddingModel()

    with pytest.raises(ValueError, match="cannot be empty"):
        model.encode([])

def test_encode_rejects_empty_text(mock_model):
    model = EmbeddingModel()

    with pytest.raises(ValueError, match="non-empty strings"):
        model.encode(["Valid text", "   "])  # Second text is empty after stripping whitespace

def test_encode_query_rejects_empty_query(mock_model):
    model = EmbeddingModel()

    with pytest.raises(ValueError, match="non-empty"):
        model.encode_query("   ")  # Query is empty after stripping whitespace