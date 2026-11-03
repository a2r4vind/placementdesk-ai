import numpy as np
import pytest 

from backend.app.retrieval.similarity import (
    normalize_vectors,
    cosine_similarity,
)

def test_normalize_single_vector():
    vector = np.array([3.0, 4.0], dtype=np.float32)

    result = normalize_vectors(vector)

    assert result.shape == (2,)
    assert result.dtype == np.float32
    assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-6)

def test_normalize_multiple_vectors():
    vectors = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ],
        dtype=np.float32,
    )

    result = normalize_vectors(vectors)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(
        np.linalg.norm(result, axis=1),
        np.ones(2),
        atol=1e-6,
    )

def test_normalize_rejects_zero_vector():
    vector = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="Zero vectors cannot be normalized."):
        normalize_vectors(vector)

def test_normalize_rejects_non_finite_values():
    vector = np.array([1.0, np.nan, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite values"):
        normalize_vectors(vector)

def test_cosine_similarity_identical_vectors():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    documents = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    scores = cosine_similarity(query, documents)

    np.testing.assert_allclose(scores, [1.0], atol=1e-6)

def test_cosine_similarity_orthogonal_vectors():
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array([[0.0, 1.0]], dtype=np.float32)

    scores = cosine_similarity(query, documents)

    np.testing.assert_allclose(scores, [0.0], atol=1e-6)

def test_cosine_similarity_opposite_vectors():
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array([[-1.0, 0.0]], dtype=np.float32)

    scores = cosine_similarity(query, documents)

    np.testing.assert_allclose(scores, [-1.0], atol=1e-6)

def test_cosine_similarity_multiple_documents():
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    scores = cosine_similarity(query, documents)

    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, [1.0, 0.0, -1.0], atol=1e-6)

def test_cosine_similarity_rejects_dimension_mismatch():
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Query and document dimensions must match."):
        cosine_similarity(query, documents)

def test_cosine_similarity_rejects_zero_vector():
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array([[0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Cosine similarity is undefined for zero vectors."):
        cosine_similarity(query, documents)

def test_cosine_similarity_rejects_non_finite_values():
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array([[np.nan, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite values"):
        cosine_similarity(query, documents)
