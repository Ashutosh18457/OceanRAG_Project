"""
MODULE 11: FINAL OUTPUT & MAIN ORCHESTRATION
=============================================
GT-RAG (Geo-Temporal Retrieval Augmented Generation)
Complete pipeline for oceanographic data analysis.

This module integrates all components:
- Data Loading
- Preprocessing
- Embedding Generation
- Vector Database
- Query Processing
- Retrieval Engine
- Trend Analysis
- Visualization
- LLM Response Generation

Returns:
- Analytical result
- Natural language explanation
- Visualization graphs
"""

import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple
import json

# Import all modules
from data_loader import OceanDataLoader, create_sample_data
from preprocessing import OceanDataPreprocessor
from embedding import OceanEmbeddingGenerator
from vector_db import ChromaVectorDB, OceanVectorStore
from query_processor import OceanQueryProcessor, QueryToEmbedding
from retrieval_engine import OceanRetrievalEngine, SemanticRetrieval
from trend_analysis import OceanTrendAnalyzer
from visualization import OceanVisualizer, InteractiveOceanVisualizer


class GTRAGSystem:
    """
    Main GT-RAG System that orchestrates all components.
    """
    
    def __init__(self, data_path: Optional[str] = None,
                 persist_dir: str = "./ocean_chroma"):
        """
        Initialize the GT-RAG system.
        
        Args:
            data_path: Path to NETCDF data file
            persist_dir: Directory for vector database persistence
        """
        print("=" * 60)
        print("GT-RAG: Geo-Temporal Retrieval Augmented Generation")
        print("=" * 60)
        
        # Configuration
        self.data_path = data_path or "sample_ocean_data.nc"
        self.persist_dir = persist_dir
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.embedding_generator = None
        self.vector_db = None
        self.vector_store = None
        self.query_processor = None
        self.retrieval_engine = None
        self.trend_analyzer = None
        self.visualizer = None
        
        # Data storage
        self.records = []
        self.chunks = []
        self.is_indexed = False
        
    def initialize(self, create_sample: bool = True):
        """
        Initialize all system components.
        
        Args:
            create_sample: Whether to create sample data if no file exists
        """
        print("\n[1/7] Initializing components...")
        
        # Create sample data if needed
        if create_sample and not os.path.exists(self.data_path):
            print(f"  Creating sample data: {self.data_path}")
            create_sample_data(self.data_path)
        
        # Initialize data loader
        self.data_loader = OceanDataLoader(self.data_path)
        
        # Initialize preprocessor
        self.preprocessor = None
        
        # Initialize embedding generator
        self.embedding_generator = OceanEmbeddingGenerator()
        
        # Initialize vector database
        self.vector_db = ChromaVectorDB(
            collection_name="ocean_data",
            persist_directory=self.persist_dir
        )
        self.vector_store = OceanVectorStore(
            self.vector_db, 
            self.embedding_generator
        )
        
        # Initialize query processor
        self.query_processor = OceanQueryProcessor()
        
        # Initialize retrieval engine
        self.retrieval_engine = OceanRetrievalEngine(
            self.vector_db,
            self.embedding_generator
        )
        
        # Initialize trend analyzer
        self.trend_analyzer = OceanTrendAnalyzer()
        
        # Initialize visualizer
        self.visualizer = OceanVisualizer()
        
        print("  ✓ All components initialized")
        
    def load_and_process_data(self, max_records: int = 10000):
        """
        Load and preprocess oceanographic data.
        
        Args:
            max_records: Maximum number of records to generate
        """
        print("\n[2/7] Loading and processing data...")
        
        # Load dataset
        self.data_loader.load_dataset()
        
        # Extract variables
        variables = self.data_loader.extract_variables()
        coordinates = self.data_loader.get_coordinates()
        
        # Initialize preprocessor with extracted data
        self.preprocessor = OceanDataPreprocessor(variables, coordinates)
        
        # Generate records
        records = self.preprocessor.generate_records(max_points=max_records)
        
        # Create chunks
        chunks = self.preprocessor.chunk_records(chunk_size=100)
        
        print(f"  ✓ Generated {len(records)} records in {len(chunks)} chunks")
        
    def build_index(self):
        """
        Build vector index from processed data.
        """
        print("\n[3/7] Building vector index...")
        
        if not self.preprocessor or not self.preprocessor.records:
            raise ValueError("No data to index. Call load_and_process_data() first.")
        
        # Index records
        n_records = self.vector_store.index_records(self.preprocessor.records)
        
        # Index chunks for chunk-level retrieval
        n_chunks = self.vector_store.index_chunks(self.preprocessor.chunks)
        
        self.is_indexed = True
        print(f"  ✓ Indexed {n_records} records and {n_chunks} chunks")
        
    def query(self, user_query: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Process a user query and return results.
        
        Args:
            user_query: Natural language query
            n_results: Number of results to retrieve
            
        Returns:
            Dictionary with query results and analysis
        """
        if not self.is_indexed:
            raise ValueError("Data not indexed. Call build_index() first.")
        
        print(f"\n[QUERY] {user_query}")
        
        # Step 1: Process query
        extracted = self.query_processor.process(user_query)
        print(f"  Extracted: {self.query_processor.format_extracted()}")
        
        # Step 2: Generate query embedding
        query_to_emb = QueryToEmbedding(self.embedding_generator)
        query_embedding = query_to_emb.create_query_embedding(
            user_query, extracted
        )
        
        # Step 3: Retrieve relevant data
        results = self.retrieval_engine.retrieve(
            query_embedding=query_embedding,
            n_results=n_results,
            parameter_filter=extracted.get('parameter'),
            location_filter=extracted.get('location'),
            time_filter=extracted.get('time_period')
        )
        
        # Step 4: Get statistics
        stats = self.retrieval_engine.get_statistics(results)
        
        # Step 5: Perform trend analysis if enough data
        analysis = None
        if len(results) >= 5:
            analysis = self.trend_analyzer.analyze_parameter(
                [r['metadata'] for r in results],
                extracted.get('parameter', 'temperature')
            )
        
        return {
            'query': user_query,
            'extracted': extracted,
            'results': results,
            'statistics': stats,
            'analysis': analysis
        }
    
    def generate_response(self, query_result: Dict[str, Any]) -> str:
        """
        Generate natural language explanation from query results.
        
        Args:
            query_result: Results from query()
            
        Returns:
            Natural language explanation
        """
        stats = query_result.get('statistics', {})
        extracted = query_result.get('extracted', {})
        analysis = query_result.get('analysis', {})
        
        lines = []
        lines.append("=" * 50)
        lines.append("ANALYSIS RESULTS")
        lines.append("=" * 50)
        
        # Parameter info
        param = extracted.get('parameter', 'ocean data')
        lines.append(f"\n📊 Parameter: {param.upper()}")
        
        # Statistics
        lines.append(f"\n📈 Statistical Summary:")
        lines.append(f"   • Count: {stats.get('count', 0)} measurements")
        lines.append(f"   • Mean: {stats.get('mean_value', 0):.2f}")
        lines.append(f"   • Std Dev: {stats.get('std_value', 0):.2f}")
        lines.append(f"   • Range: [{stats.get('min_value', 0):.2f}, {stats.get('max_value', 0):.2f}]")
        
        # Location info
        if extracted.get('location'):
            loc = extracted['location']
            lines.append(f"\n📍 Location: {loc['latitude']:.1f}°N, {loc['longitude']:.1f}°E")
        
        # Time info
        if extracted.get('time_period'):
            tp = extracted['time_period']
            if 'year' in tp:
                lines.append(f"\n⏰ Time: Year {tp['year']}")
            elif 'start_year' in tp:
                lines.append(f"\n⏰ Time: {tp['start_year']} - {tp['end_year']}")
        
        # Trend analysis
        if analysis and 'trend' in analysis:
            trend = analysis['trend']
            if trend.get('slope', 0) != 0:
                lines.append(f"\n📉 Trend Analysis:")
                lines.append(f"   • Direction: {trend.get('trend_direction', 'unknown')}")
                lines.append(f"   • Rate: {abs(trend.get('slope', 0)):.4f} per time unit")
                lines.append(f"   • R²: {trend.get('r_squared', 0):.3f}")
                lines.append(f"   • Significant: {'Yes' if trend.get('is_significant') else 'No'}")
        
        # Anomalies
        if analysis and 'anomalies' in analysis:
            n_anom = analysis['anomalies'].get('n_anomalies', 0)
            if n_anom > 0:
                lines.append(f"\n⚠️  Anomalies: {n_anom} detected")
        
        # Similarity
        lines.append(f"\n🎯 Average Retrieval Similarity: {stats.get('avg_similarity', 0):.3f}")
        
        return "\n".join(lines)
    
    def visualize_results(self, query_result: Dict[str, Any],
                         save_dir: str = "./output") -> Dict[str, str]:
        """
        Generate visualizations from query results.
        
        Args:
            query_result: Results from query()
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of saved plot paths
        """
        print("\n[VISUALIZATION] Generating plots...")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = {}
        results = query_result.get('results', [])
        
        if not results:
            return saved_paths
        
        # Extract data for visualization
        times = []
        values = []
        lats = []
        lons = []
        
        for r in results:
            meta = r.get('metadata', {})
            time_val = meta.get('time', 0)
            value_val = meta.get('value', 0)
            lat_val = meta.get('latitude', 0)
            lon_val = meta.get('longitude', 0)
            
            # Only add valid numeric values
            if (time_val is not None and value_val is not None and
                lat_val is not None and lon_val is not None):
                try:
                    times.append(float(time_val))
                    values.append(float(value_val))
                    lats.append(float(lat_val))
                    lons.append(float(lon_val))
                except (ValueError, TypeError):
                    continue
        
        times = np.array(times)
        values = np.array(values)
        lats = np.array(lats)
        lons = np.array(lons)
        
        # Temperature vs Time plot
        if len(times) > 1:
            fig1 = self.visualizer.plot_temperature_time(
                times, values,
                title=f"Ocean {query_result.get('extracted', {}).get('parameter', 'data')} Over Time",
                save_path=os.path.join(save_dir, "temperature_time.png")
            )
            import matplotlib.pyplot as plt
            plt.close(fig1)
            saved_paths['temp_time'] = os.path.join(save_dir, "temperature_time.png")
        
        # Heatmap
        if len(lats) > 1 and len(lons) > 1:
            fig2 = self.visualizer.plot_heatmap(
                lats, lons, values,
                title="Ocean Data Heatmap",
                save_path=os.path.join(save_dir, "heatmap.png")
            )
            plt.close(fig2)
            saved_paths['heatmap'] = os.path.join(save_dir, "heatmap.png")
        
        # Statistics bar chart
        stats = query_result.get('statistics', {})
        fig3 = self.visualizer.plot_statistics(
            stats,
            title="Statistical Summary",
            save_path=os.path.join(save_dir, "statistics.png")
        )
        plt.close(fig3)
        saved_paths['statistics'] = os.path.join(save_dir, "statistics.png")
        
        print(f"  ✓ Saved {len(saved_paths)} visualizations")
        
        return saved_paths
    
    def run_full_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Run the complete GT-RAG pipeline.
        
        Args:
            query: Natural language query
            
        Returns:
            Complete results with analysis, explanation, and visualizations
        """
        print("\n" + "=" * 60)
        print("RUNNING FULL GT-RAG PIPELINE")
        print("=" * 60)
        
        # Initialize if needed
        if self.data_loader is None:
            self.initialize()
            self.load_and_process_data()
            self.build_index()
        
        # Process query
        query_result = self.query(query)
        
        # Generate explanation
        explanation = self.generate_response(query_result)
        
        # Generate visualizations
        visualizations = self.visualize_results(query_result)
        
        return {
            'query': query,
            'extracted_info': query_result['extracted'],
            'results': query_result['results'],
            'statistics': query_result['statistics'],
            'analysis': query_result.get('analysis'),
            'explanation': explanation,
            'visualizations': visualizations
        }
    
    def interactive_mode(self):
        """
        Run in interactive query mode.
        """
        print("\n" + "=" * 60)
        print("GT-RAG INTERACTIVE MODE")
        print("=" * 60)
        print("Enter queries about oceanographic data.")
        print("Type 'quit' or 'exit' to end.\n")
        
        # Initialize if needed
        if self.data_loader is None:
            self.initialize()
            self.load_and_process_data()
            self.build_index()
        
        while True:
            try:
                query = input("\nQuery> ")
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query.strip():
                    continue
                
                result = self.run_full_pipeline(query)
                print(result['explanation'])
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.data_loader:
            self.data_loader.close()
        print("\n✓ Cleanup complete")


def demo():
    """Run a demonstration of the GT-RAG system."""
    print("\n" + "#" * 60)
    print("# GT-RAG SYSTEM DEMONSTRATION")
    print("#" * 60)
    
    # Create system
    gtrag = GTRAGSystem(data_path="sample_ocean_data.nc")
    
    # Initialize
    gtrag.initialize()
    
    # Load and process data
    gtrag.load_and_process_data(max_records=5000)
    
    # Build index
    gtrag.build_index()
    
    # Run example queries
    test_queries = [
        "What is the temperature in the ocean?",
        "Show me salinity data",
        "Ocean temperature trends over time"
    ]
    
    for query in test_queries:
        result = gtrag.run_full_pipeline(query)
        print(result['explanation'])
        print("\n" + "-" * 40)
    
    # Cleanup
    gtrag.cleanup()
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()

