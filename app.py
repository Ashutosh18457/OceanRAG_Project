"""
GT-RAG Streamlit Web Interface
==============================
Interactive web UI for oceanographic data analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import time
from typing import Dict, Any, Optional

# Import the main GT-RAG system
from main import GTRAGSystem

# Page configuration
st.set_page_config(
    page_title="GT-RAG Ocean Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
    }
    .warning-box {
        padding: 1rem;
        background-color: #FFF3E0;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
    }
    .stat-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'gtrag_system' not in st.session_state:
        st.session_state.gtrag_system = None
    
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False
    
    if 'is_indexed' not in st.session_state:
        st.session_state.is_indexed = False
    
    if 'query_results' not in st.session_state:
        st.session_state.query_results = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def initialize_system(progress_bar=None):
    """
    Initialize the GT-RAG system.
    
    Args:
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        Initialized GTRAGSystem
    """
    if progress_bar:
        progress_bar.progress(10)
    
    # Create system instance
    system = GTRAGSystem(
        data_path="sample_ocean_data.nc",
        persist_dir="./ocean_chroma"
    )
    
    if progress_bar:
        progress_bar.progress(30)
    
    # Initialize components
    system.initialize(create_sample=True)
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Load and process data
    system.load_and_process_data(max_records=5000)
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Build index
    system.build_index()
    
    if progress_bar:
        progress_bar.progress(100)
    
    return system


def display_statistics_summary(stats: Dict[str, Any]):
    """
    Display statistics in a nice grid layout.
    
    Args:
        stats: Statistics dictionary
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('count', 0)}</div>
            <div class="stat-label">Measurements</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('mean_value', 0):.2f}</div>
            <div class="stat-label">Mean</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats.get('std_value', 0):.2f}</div>
            <div class="stat-label">Std Dev</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        min_val = stats.get('min_value', 0)
        max_val = stats.get('max_value', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{min_val:.2f} - {max_val:.2f}</div>
            <div class="stat-label">Range</div>
        </div>
        """, unsafe_allow_html=True)


def display_trend_analysis(analysis: Dict[str, Any]):
    """
    Display trend analysis results.
    
    Args:
        analysis: Trend analysis dictionary
    """
    if not analysis or 'trend' not in analysis:
        st.info("Not enough data for trend analysis")
        return
    
    trend = analysis['trend']
    
    st.markdown("### 📈 Trend Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        direction = trend.get('trend_direction', 'unknown')
        emoji = "📈" if direction == "increasing" else "📉" if direction == "decreasing" else "➡️"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{emoji} {direction.title()}</div>
            <div class="stat-label">Trend Direction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        slope = trend.get('slope', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{abs(slope):.4f}</div>
            <div class="stat-label">Rate of Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        r_squared = trend.get('r_squared', 0)
        significant = trend.get('is_significant', False)
        sig_text = "Yes ✓" if significant else "No ✗"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{r_squared:.3f}</div>
            <div class="stat-label">R² (Significant: {sig_text})</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Anomalies section
    if 'anomalies' in analysis:
        n_anom = analysis['anomalies'].get('n_anomalies', 0)
        if n_anom > 0:
            st.warning(f"⚠️ {n_anom} anomalies detected in the data")
        else:
            st.success("✓ No anomalies detected")


def display_results_table(results: list):
    """
    Display retrieval results in a table.
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        st.info("No results to display")
        return
    
    # Convert to DataFrame for display
    data = []
    for i, r in enumerate(results):
        meta = r.get('metadata', {})
        data.append({
            'Rank': i + 1,
            'Parameter': meta.get('parameter', 'N/A'),
            'Value': f"{meta.get('value', 0):.4f}",
            'Latitude': f"{meta.get('latitude', 0):.2f}",
            'Longitude': f"{meta.get('longitude', 0):.2f}",
            'Time': f"{meta.get('time', 0):.1f}",
            'Similarity': f"{r.get('similarity', 0):.3f}"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_visualizations(query_result: Dict[str, Any], save_dir: str = "./output"):
    """
    Display visualizations from query results.
    
    Args:
        query_result: Query result dictionary
        save_dir: Directory to save plots
    """
    results = query_result.get('results', [])
    
    if not results:
        st.info("No data available for visualization")
        return
    
    # Extract data
    times = []
    values = []
    lats = []
    lons = []
    
    for r in results:
        meta = r.get('metadata', {})
        times.append(float(meta.get('time', 0)))
        values.append(float(meta.get('value', 0)))
        lats.append(float(meta.get('latitude', 0)))
        lons.append(float(meta.get('longitude', 0)))
    
    times = np.array(times)
    values = np.array(values)
    lats = np.array(lats)
    lons = np.array(lons)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🗺️ Heatmap", "📊 Statistics"])
    
    with tab1:
        if len(times) > 1:
            # Create time series plot
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, values, 'b-', linewidth=1.5, alpha=0.8)
            ax.fill_between(times, values, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(times, values, 1)
            p = np.poly1d(z)
            ax.plot(times, p(times), 'r--', linewidth=2, label=f'Trend: {z[0]:.4f}')
            
            ax.set_xlabel('Time (days)', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Ocean Data Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.info("Not enough data points for time series")
    
    with tab2:
        if len(lats) > 1 and len(lons) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Reshape values if needed
            side = int(np.sqrt(len(values)))
            if side * side <= len(values):
                values_2d = values[:side*side].reshape(side, side)
            else:
                # Fallback: create a simple 2D grid
                min_dim = min(len(lats), len(lons))
                values_2d = values[:min_dim*min_dim].reshape(min_dim, min_dim)
            
            im = ax.imshow(values_2d, extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          aspect='auto', cmap='RdYlBu_r', origin='lower')
            
            ax.set_xlabel('Longitude (°)', fontsize=12)
            ax.set_ylabel('Latitude (°)', fontsize=12)
            ax.set_title('Ocean Data Heatmap', fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value', fontsize=12)
            
            st.pyplot(fig)
        else:
            st.info("Not enough data points for heatmap")
    
    with tab3:
        stats = query_result.get('statistics', {})
        fig, ax = plt.subplots(figsize=(10, 5))
        
        stat_names = ['mean', 'std', 'min', 'max']
        stat_values = [stats.get(s, 0) for s in stat_names]
        
        bars = ax.bar(stat_names, stat_values, color=['#1E88E5', '#43A047', '#E53935', '#FB8C00'])
        
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Statistical Summary', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, stat_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig)


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("🌊 GT-RAG System")
    st.sidebar.markdown("---")
    
    # Data management section in sidebar
    st.sidebar.subheader("📁 Data Management")
    
    if st.sidebar.button("🔄 Initialize System", type="primary"):
        with st.spinner("Initializing GT-RAG system..."):
            progress_bar = st.progress(0)
            st.session_state.gtrag_system = initialize_system(progress_bar)
            st.session_state.is_initialized = True
            st.session_state.is_indexed = True
            st.success("✓ System initialized and data indexed!")
    
    # Status indicators
    st.sidebar.markdown("### System Status")
    
    if st.session_state.is_initialized:
        st.sidebar.success("✓ System Initialized")
    else:
        st.sidebar.warning("○ System Not Initialized")
    
    if st.session_state.is_indexed:
        st.sidebar.success("✓ Data Indexed")
    else:
        st.sidebar.warning("○ Data Not Indexed")
    
    # Query history
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Query History")
    
    if st.session_state.query_history:
        for i, (query, time_str) in enumerate(reversed(st.session_state.query_history[-5:])):
            st.sidebar.text(f"{i+1}. {query[:30]}...")
    else:
        st.sidebar.info("No queries yet")
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.session_state.query_results = None
        st.rerun()
    
    # Main content
    st.markdown('<div class="main-header">🌊 GT-RAG Oceanographic Data Analysis</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the GT-RAG (Geo-Temporal Retrieval Augmented Generation) system! 
    Ask questions about oceanographic data in natural language.
    """)
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - What is the temperature in the ocean?
        - Show me salinity data
        - Ocean temperature trends over time
        - Temperature at latitude 10, longitude 50
        - Salinity from year 2020 to 2025
        """)
    
    st.markdown("---")
    
    # Query input section
    st.subheader("🔍 Query Input")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the ocean temperature trend?",
            disabled=not st.session_state.is_initialized
        )
    
    with col2:
        n_results = st.selectbox(
            "Results:",
            options=[5, 10, 15, 20],
            index=1,
            disabled=not st.session_state.is_initialized
        )
    
    submit_button = st.button(
        "🚀 Submit Query",
        type="primary",
        disabled=not st.session_state.is_initialized or not query
    )
    
    # Process query
    if submit_button and query:
        with st.spinner("Processing query..."):
            try:
                # Get results
                result = st.session_state.gtrag_system.run_full_pipeline(query)
                
                # Store in session
                st.session_state.query_results = result
                st.session_state.query_history.append((query, time.strftime("%H:%M:%S")))
                
                st.success("✓ Query processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display results
    if st.session_state.query_results:
        result = st.session_state.query_results
        
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Query Results</div>', unsafe_allow_html=True)
        
        # Query interpretation
        with st.expander("🔎 Query Interpretation", expanded=True):
            extracted = result.get('extracted_info', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                param = extracted.get('parameter', 'Not specified')
                st.info(f"**Parameter:** {param}")
            
            with col2:
                if extracted.get('location'):
                    loc = extracted['location']
                    st.info(f"**Location:** {loc['latitude']:.1f}°, {loc['longitude']:.1f}°")
                else:
                    st.info("**Location:** Not specified")
            
            with col3:
                if extracted.get('time_period'):
                    tp = extracted['time_period']
                    time_str = tp.get('year', '') or f"{tp.get('start_year', '')}-{tp.get('end_year', '')}"
                    st.info(f"**Time:** {time_str}")
                else:
                    st.info("**Time:** Not specified")
        
        # Statistics summary
        st.markdown("### 📈 Statistical Summary")
        stats = result.get('statistics', {})
        display_statistics_summary(stats)
        
        # Trend analysis
        if result.get('analysis'):
            st.markdown("---")
            display_trend_analysis(result['analysis'])
        
        # Results table
        st.markdown("---")
        st.markdown("### 📋 Retrieved Data")
        display_results_table(result.get('results', []))
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📊 Visualizations")
        display_visualizations(result)
        
        # Natural language explanation
        st.markdown("---")
        st.markdown("### 💬 Analysis Explanation")
        explanation = result.get('explanation', '')
        st.markdown(f"```\n{explanation}\n```")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #757575;'>
        GT-RAG System | Oceanographic Data Analysis | Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

