"""
MODULE 6: RETRIEVAL ENGINE
==========================
Performs semantic vector search in the database to retrieve
the most relevant data chunks.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class OceanRetrievalEngine:
    """
    Retrieval engine for semantic search of oceanographic data.
    """
    
    def __init__(self, vector_db, embedding_generator):
        """
        Initialize the retrieval engine.
        
        Args:
            vector_db: ChromaVectorDB instance
            embedding_generator: Embedding generator instance
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.last_results = []
        
    def retrieve(self, query_embedding: np.ndarray,
                n_results: int = 10,
                parameter_filter: Optional[str] = None,
                location_filter: Optional[Dict[str, float]] = None,
                time_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant data chunks.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to retrieve
            parameter_filter: Filter by parameter
            location_filter: Filter by geographic location
            time_filter: Filter by time period
            
        Returns:
            List of retrieved results
        """
        # Build where clause for metadata filtering
        where_clause = None
        if parameter_filter:
            where_clause = {'parameter': parameter_filter}
        
        # Search in vector DB
        results = self.vector_db.search(
            query_embedding=query_embedding,
            n_results=n_results * 2,  # Get more for filtering
            where=where_clause
        )
        
        # Process results
        retrieved = []
        for i in range(len(results['ids'])):
            result = {
                'id': results['ids'][i],
                'distance': results['distances'][i],
                'similarity': 1 - results['distances'][i],  # Convert distance to similarity
                'metadata': results['metadatas'][i],
                'document': results['documents'][i]
            }
            retrieved.append(result)
        
        # Apply additional filters
        if location_filter or time_filter:
            retrieved = self._apply_filters(
                retrieved, location_filter, time_filter
            )[:n_results]
        else:
            retrieved = retrieved[:n_results]
        
        self.last_results = retrieved
        return retrieved
    
    def _apply_filters(self, results: List[Dict[str, Any]],
                      location_filter: Optional[Dict[str, float]],
                      time_filter: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply geo-temporal filters to results.
        
        Args:
            results: Initial search results
            location_filter: Location filter criteria
            time_filter: Time filter criteria
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            meta = result.get('metadata', {})
            keep = True
            
            # Location filter
            if location_filter:
                lat = meta.get('latitude', 0)
                lon = meta.get('longitude', 0)
                
                lat_tol = location_filter.get('latitude_tolerance', 5.0)
                lon_tol = location_filter.get('longitude_tolerance', 5.0)
                
                target_lat = location_filter.get('latitude', lat)
                target_lon = location_filter.get('longitude', lon)
                
                if abs(lat - target_lat) > lat_tol or abs(lon - target_lon) > lon_tol:
                    keep = False
            
            # Time filter
            if keep and time_filter:
                time_val = meta.get('time', '')
                if time_val:
                    try:
                        time_num = float(time_val)
                        
                        if 'start_time' in time_filter:
                            if time_num < time_filter['start_time']:
                                keep = False
                        if 'end_time' in time_filter:
                            if time_num > time_filter['end_time']:
                                keep = False
                    except:
                        pass
            
            if keep:
                filtered.append(result)
        
        return filtered
    
    def get_top_results(self, results: List[Dict[str, Any]], 
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-k results by similarity.
        
        Args:
            results: All results
            top_k: Number of top results
            
        Returns:
            Top-k results
        """
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:top_k]
    
    def group_by_parameter(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group results by parameter type.
        
        Args:
            results: List of results
            
        Returns:
            Dictionary of parameter -> results list
        """
        grouped = {}
        for result in results:
            param = result.get('metadata', {}).get('parameter', 'unknown')
            if param not in grouped:
                grouped[param] = []
            grouped[param].append(result)
        return grouped
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about retrieved results.
        
        Args:
            results: List of results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {'count': 0}
        
        # Extract values
        values = [r.get('metadata', {}).get('value', 0) for r in results]
        values = [v for v in values if v is not None]
        
        stats = {
            'count': len(results),
            'mean_value': float(np.mean(values)) if values else 0,
            'std_value': float(np.std(values)) if values else 0,
            'min_value': float(np.min(values)) if values else 0,
            'max_value': float(np.max(values)) if values else 0,
            'avg_similarity': float(np.mean([r['similarity'] for r in results]))
        }
        
        # Group by parameter
        grouped = self.group_by_parameter(results)
        stats['parameters'] = {k: len(v) for k, v in grouped.items()}
        
        return stats
    
    def format_results(self, results: List[Dict[str, Any]], 
                      max_display: int = 5) -> str:
        """
        Format results as readable string.
        
        Args:
            results: Results to format
            max_display: Maximum number to display
            
        Returns:
            Formatted string
        """
        if not results:
            return "No results found."
        
        lines = []
        lines.append(f"Retrieved {len(results)} results:")
        lines.append("-" * 50)
        
        for i, result in enumerate(results[:max_display]):
            meta = result.get('metadata', {})
            lines.append(f"\n{i+1}. {meta.get('parameter', 'unknown').title()}")
            lines.append(f"   Value: {meta.get('value', 'N/A')}")
            lines.append(f"   Location: {meta.get('latitude', 0):.1f}°, {meta.get('longitude', 0):.1f}°")
            lines.append(f"   Depth: {meta.get('depth', 0):.0f}m")
            lines.append(f"   Similarity: {result.get('similarity', 0):.3f}")
        
        if len(results) > max_display:
            lines.append(f"\n... and {len(results) - max_display} more results")
        
        return "\n".join(lines)


class SemanticRetrieval:
    """
    High-level semantic retrieval with query expansion.
    """
    
    def __init__(self, retrieval_engine: OceanRetrievalEngine):
        """
        Initialize semantic retrieval.
        
        Args:
            retrieval_engine: OceanRetrievalEngine instance
        """
        self.retrieval_engine = retrieval_engine
        
    def expand_query(self, query: str, extracted_info: Dict[str, Any]) -> str:
        """
        Expand query with extracted context.
        
        Args:
            query: Original query
            extracted_info: Extracted query information
            
        Returns:
            Expanded query string
        """
        expansions = [query]
        
        # Add related terms
        param = extracted_info.get('parameter')
        if param == 'temperature':
            expansions.extend(['ocean heat', 'sea surface temperature', 'thermal'])
        elif param == 'salinity':
            expansions.extend(['salt content', 'ocean salinity', 'seawater composition'])
        
        # Combine
        expanded = " OR ".join(expansions)
        return expanded
    
    def retrieve_with_context(self, query_embedding: np.ndarray,
                             extracted_info: Dict[str, Any],
                             n_results: int = 10) -> Dict[str, Any]:
        """
        Retrieve with expanded context.
        
        Args:
            query_embedding: Query vector
            extracted_info: Extracted query information
            n_results: Number of results
            
        Returns:
            Retrieval results with metadata
        """
        # Retrieve
        results = self.retrieval_engine.retrieve(
            query_embedding=query_embedding,
            n_results=n_results,
            parameter_filter=extracted_info.get('parameter')
        )
        
        # Get statistics
        stats = self.retrieval_engine.get_statistics(results)
        
        # Group by parameter
        grouped = self.retrieval_engine.group_by_parameter(results)
        
        return {
            'results': results,
            'statistics': stats,
            'grouped_by_parameter': grouped,
            'extracted_info': extracted_info
        }


if __name__ == "__main__":
    # Test retrieval
    print("Testing OceanRetrievalEngine...")
    
    from vector_db import ChromaVectorDB
    from embedding import OceanEmbeddingGenerator
    
    # Setup
    generator = OceanEmbeddingGenerator()
    db = ChromaVectorDB(collection_name="test_retrieval")
    
    # Add some test data
    test_docs = [
        "Temperature 25C at tropical ocean",
        "Temperature 15C at temperate region",
        "Salinity 35 PSU in ocean waters",
        "Temperature 20C at depth 100m"
    ]
    
    embeddings = generator.generate_embeddings(test_docs)
    
    for i, (emb, doc) in enumerate(zip(embeddings, test_docs)):
        param = 'temperature' if 'temperature' in doc.lower() else 'salinity'
        meta = {'parameter': param, 'value': 25 if param == 'temperature' else 35}
        
        db.add_vectors(
            ids=[f"test_{i}"],
            embeddings=[emb.tolist()],
            metadatas=[meta],
            documents=[doc]
        )
    
    # Test retrieval
    engine = OceanRetrievalEngine(db, generator)
    query_emb = generator.generate_embedding("ocean temperature measurements")
    results = engine.retrieve(query_emb, n_results=3)
    
    print(engine.format_results(results))
    print(f"\nStatistics: {engine.get_statistics(results)}")

