import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class TextEmbeddingPreprocessor(BaseEstimator, TransformerMixin):
    """
    Converts unstructured string/text columns into dense numerical embeddings
    using a lightweight pre-trained language model.
    Applies PCA to shrink the embedding dimensions so they fit within the 
    PFN's 20-feature maximum limit.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', n_components=5):
        self.model_name = model_name
        self.n_components = n_components
        self.model = None
        self.pcas = {} # One PCA per text column
        
    def fit(self, X, y=None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Run `uv pip install sentence-transformers`")
            
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            
        X_df = pd.DataFrame(X)
        self.pcas = {}
        
        for col in X_df.columns:
            # We fit a PCA for each text column to compress it
            texts = X_df[col].astype(str).tolist()
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # If we have very few samples, we can't extract n_components
            n_comp = min(self.n_components, embeddings.shape[0], embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit(embeddings)
            
            self.pcas[col] = pca
            
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("TextEmbeddingPreprocessor is not fitted yet.")
            
        X_df = pd.DataFrame(X)
        transformed_cols = []
        
        for col in X_df.columns:
            pca = self.pcas[col]
            texts = X_df[col].astype(str).tolist()
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Reduce dimension
            reduced_emb = pca.transform(embeddings)
            transformed_cols.append(reduced_emb)
            
        # Concatenate all reduced text embeddings into one big 2D array
        if transformed_cols:
            return np.hstack(transformed_cols)
        else:
            return np.empty((X.shape[0], 0))
