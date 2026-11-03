import numpy as np

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize a input vector or a matrix of vectors to unit length.

    Args:
        1D array: (embedding_dimension,)
        2D array: (number_of_vectors, embedding_dimension)
        
    Returns:
        np.ndarray: Normalized float32 array of the same shape as input.
    """
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim not in (1, 2):
        raise ValueError("Vectors must be a 1D or 2D array.")

    if vectors.size == 0:
        raise ValueError("Vectors cannot be empty.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors must contain non-finite values.")
    
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Zero vectors cannot be normalized.")

    return vectors / norms

def cosine_similarity(
        query_vector: np.ndarray,
        document_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute the cosine similarity between a query vector and a set of document vectors.

    Args:
        query_vector (np.ndarray): A 1D array (shape: (embedding_dimension,)) representing the query vector.
        document_vectors (np.ndarray): A 2D array (shape: (number_of_document_vectors, embedding_dimension)) where each row is a document vector.

    Returns:
        np.ndarray: A 1D float32 array of cosine similarity scores between the query and each document vector.
    """
    query_vector = np.asarray(query_vector, dtype=np.float32)
    document_vectors = np.asarray(document_vectors, dtype=np.float32)
    
    if query_vector.ndim != 1:
        raise ValueError("Query vector must be a 1D array.")

    if document_vectors.ndim != 2:
        raise ValueError("Document vectors must be a 2D array.")
    
    if query_vector.size == 0:
        raise ValueError("Query vector cannot be empty.")

    if document_vectors.shape[0] == 0:
        raise ValueError("Document vectors cannot be empty.")

    if query_vector.shape[0] != document_vectors.shape[1]:
        raise ValueError("Query and document dimensions must match.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector must contain non-finite values.")

    if not np.isfinite(document_vectors).all():
        raise ValueError("Document vectors must contain non-finite values.")

    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    scores = np.dot(document_vectors, query_vector)
    scores = scores / (document_norms * query_norm)

    # Protecting against tiny floating-point deviations outside [-1, 1].
    return np.clip(scores, -1.0, 1.0).astype(np.float32)