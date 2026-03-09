"""
MODULE 3: EMBEDDING GENERATION
==============================
Uses sentence transformer models to convert ocean data descriptions
into vector embeddings for semantic search.
"""

import numpy as np
from typing import List, Dict, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Try to import sentence_transformers, install if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ sentence-transformers not installed. Using fallback embeddings.")


class OceanEmbeddingGenerator:
    """
    Generates vector embeddings for oceanographic data descriptions
    using sentence transformer models.
    """
    
    # Default model - lightweight and effective for scientific text
    DEFAULT_MODEL = 'all-MiniLM-L6-v2'
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of sentence-transformer model to use
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self.embedding_dim = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"✓ Loaded embedding model: {self.model_name}")
                print(f"  Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                print(f"⚠ Failed to load model {self.model_name}: {e}")
                self.model = None
        else:
            print("⚠ Using fallback random embeddings (install sentence-transformers for real embeddings)")
            self.embedding_dim = 384  # Standard dimension for MiniLM
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text description.
        
        Args:
            text: Text description to embed
            
        Returns:
            Vector embedding array
        """
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        else:
            # Fallback: generate deterministic random embedding based on text
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def generate_embeddings(self, texts: List[str], 
                           batch_size: int = 32,
                           show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple text descriptions.
        
        Args:
            texts: List of text descriptions
            batch_size: Batch size for processing
            show_progress: Show progress indicator
            
        Returns:
            Array of embeddings (num_texts x embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self.model is not None:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        else:
            # Fallback embeddings
            embeddings = []
            for i, text in enumerate(texts):
                if show_progress and i % 100 == 0:
                    print(f"  Processing {i}/{len(texts)}...")
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.embedding_dim).astype(np.float32)
                # Normalize
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
            return np.array(embeddings)
    
    def embed_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to records.
        
        Args:
            records: List of data records
            
        Returns:
            Records with embedding field added
        """
        if not records:
            return []
        
        # Extract text descriptions
        texts = [r.get('text', '') for r in records]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to records
        for i, record in enumerate(records):
            record['embedding'] = embeddings[i]
        
        print(f"✓ Generated embeddings for {len(records)} records")
        return records
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to data chunks.
        
        Args:
            chunks: List of data chunks
            
        Returns:
            Chunks with embedding field added
        """
        if not chunks:
            return []
        
        # Extract text summaries
        texts = [c.get('text_summary', '') for c in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        print(f"✓ Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))
    
    def find_similar(self, query_embedding: np.ndarray, 
                    embeddings: List[np.ndarray],
                    top_k: int = 5) -> List[int]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            Indices of top-k similar embeddings
        """
        similarities = []
        for emb in embeddings:
            sim = self.compute_similarity(query_embedding, emb)
            similarities.append(sim)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices.tolist()


class OceanQueryEmbedding:
    """
    Specialized embedding generator for queries.
    Creates rich query representations combining multiple aspects.
    """
    
    def __init__(self, embedding_generator: OceanEmbeddingGenerator):
        """
        Initialize with base embedding generator.
        
        Args:
            embedding_generator: OceanEmbeddingGenerator instance
        """
        self.generator = embedding_generator
        
    def create_query_embedding(self, query: str,
                               parameter: Optional[str] = None,
                               location: Optional[Dict[str, float]] = None,
                               time_period: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create enriched query embedding with context.
        
        Args:
            query: Original natural language query
            parameter: Extracted parameter (temperature, salinity)
            location: Location dict with lat/lon
            time_period: Time period dict
            
        Returns:
            Enriched query embedding
        """
        # Build enhanced query text
        enhanced_parts = [query]
        
        if parameter:
            enhanced_parts.append(f"Focus on {parameter} measurements")
            
        if location:
            lat = location.get('latitude', 0)
            lon = location.get('longitude', 0)
            enhanced_parts.append(f"at latitude {lat}, longitude {lon}")
            
        if time_period:
            start = time_period.get('start', '')
            end = time_period.get('end', '')
            if start or end:
                enhanced_parts.append(f"during time period {start} to {end}")
        
        enhanced_query = ". ".join(enhanced_parts) + "."
        
        # Generate embedding
        return self.generator.generate_embedding(enhanced_query)


if __name__ == "__main__":
    # Test embedding generation
    print("Testing OceanEmbeddingGenerator...")
    
    generator = OceanEmbeddingGenerator()
    
    # Test single embedding
    test_text = "The sea surface temperature is 25.5 degrees at 10.5°N, 50.2°E."
    emb = generator.generate_embedding(test_text)
    print(f"Single embedding shape: {emb.shape}")
    
    # Test batch embeddings
    texts = [
        "Ocean temperature is 20 degrees at location 10N, 50E",
        "Salinity measurement of 35 PSU at depth 100m",
        "Water pressure of 500 db at tropical region"
    ]
    embeddings = generator.generate_embeddings(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = generator.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between texts 0 and 1: {sim:.4f}")

