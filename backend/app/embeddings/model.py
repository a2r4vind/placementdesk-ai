from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """
    Generates normalized vector embeddings for documents and queries using a pre-trained SentenceTransformer model.

    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initializes the EmbeddingModel with a specified pre-trained model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model to use for generating embeddings.
                              Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            raise ValueError(f"Unable to determine embedding dimension for model '{model_name}'.")

        self.embedding_dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Generates normalized embeddings for a list of texts.

        Args:
            texts (list[str]): A list of strings to be converted into embeddings.

        Returns:
            np.ndarray: A 2D array of shape (len(texts), self.embedding_dimension) containing the normalized embeddings for each text.
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty.")

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All input texts must be non-empty strings.")

        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(f"Document embeddings must be a 2D array, but got {embeddings.ndim}D array.")

        if embeddings.shape != (len(texts), self.embedding_dimension):
            raise ValueError(
                f"Expected embeddings shape {(len(texts), self.embedding_dimension)},"
                f" but got {embeddings.shape}."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non-finite values (NaN or Inf).")

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single search query.
        
        Returns:
            np.ndarray: Shape (embedding dimension,).
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (self.embedding_dimension,):
            raise ValueError(
                f"Expected query embedding shape {(self.embedding_dimension,)},"
                f" but got {embedding.shape}."
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Query embedding contains non-finite values (NaN or Inf).")

        return embedding