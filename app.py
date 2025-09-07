"""
ChainSense - Supply Chain Risk Analyzer
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Tuple, Optional

# Import custom modules
from data_processor import SupplyChainData
from graph_analyzer import GraphAnalyzer
from visualizer import SupplyChainVisualizer
from advanced_risk_analyzer import AdvancedRiskAnalyzer

# Page configuration
st.set_page_config(
    page_title="ChainSense - Supply Chain Risk Analyzer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e1e5e9;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 0;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Risk level styling */
    .risk-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #d63031;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(214, 48, 49, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #d63031;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(253, 121, 168, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e17055;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(253, 203, 110, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00cec9;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(85, 239, 196, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card-background);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: #2c3e50 !important;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #1f77b4 !important;
    }
    
    /* Success/Info/Warning/Error styling */
    .stSuccess {
        background: linear-gradient(135deg, #00b894, #55efc4);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    .stError {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        border: none;
        border-radius: 10px;
        color: white;
    }
    
    /* Feature highlight boxes */
    .feature-box {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    /* Welcome screen styling */
    .welcome-container {
        background: var(--card-background);
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Progress indicator */
    .progress-step {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: var(--primary-color);
        color: white;
        text-align: center;
        line-height: 30px;
        margin-right: 10px;
        font-weight: bold;
    }
    
    .progress-step.completed {
        background: var(--success-color);
    }
    
    .progress-step.current {
        background: var(--warning-color);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = SupplyChainData()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'advanced_analyzer' not in st.session_state:
        st.session_state.advanced_analyzer = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'show_mapping' not in st.session_state:
        st.session_state.show_mapping = False

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔗 ChainSense</h1>
        <p>AI-powered Supply Chain Risk Analyzer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0;">🎛️ Control Panel</h2>
            <p style="margin: 0; opacity: 0.8;">Configure your analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress indicator
        progress_steps = [
            ("📁", "Upload Data", st.session_state.data_loaded),
            ("⚙️", "Configure", st.session_state.data_loaded),
            ("🚀", "Analyze", st.session_state.analysis_complete)
        ]
        
        st.markdown("**📋 Progress**")
        progress_html = ""
        for icon, step, completed in progress_steps:
            status_class = "completed" if completed else "current" if step == "Configure" and st.session_state.data_loaded else ""
            progress_html += f"""
            <div style="margin: 0.5rem 0; display: flex; align-items: center;">
                <span class="progress-step {status_class}">{icon}</span>
                <span style="margin-left: 10px; {'color: #2ca02c; font-weight: bold;' if completed else ''}">{step}</span>
                {'✅' if completed else '' if not status_class else '⏳'}
            </div>
            """
        
        st.markdown(progress_html, unsafe_allow_html=True)
        st.markdown("---")
        
        # Data upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with supply chain data. Required columns: supplier, customer"
        )
        
        # Sample data option
        if st.button("📊 Load Sample Data"):
            sample_df = st.session_state.data_processor.create_sample_data()
            st.session_state.data_processor.df = sample_df
            st.session_state.data_loaded = True
            st.success("Sample data loaded successfully!")
        
        # Process uploaded file
        if uploaded_file is not None:
            if st.button("🚀 Process File", use_container_width=True):
                if st.session_state.data_processor.load_csv(uploaded_file):
                    st.success("📁 File loaded successfully!")
                    
                    # Try automatic relationship detection
                    with st.spinner("🔍 Analyzing data structure..."):
                        if st.session_state.data_processor.auto_detect_relationships():
                            st.session_state.data_loaded = True
                            st.success("✨ Supply chain relationships detected automatically!")
                            st.rerun()
                        else:
                            # Show manual mapping interface
                            st.warning("Could not auto-detect relationships. Please use manual mapping below.")
                            st.session_state.show_mapping = True
                            st.rerun()
        
        # Analysis options
        if st.session_state.data_loaded:
            st.subheader("⚙️ Analysis Options")
            
            # Graph layout options
            layout_option = st.selectbox(
                "Graph Layout",
                ["spring", "hierarchical", "circular"],
                help="Choose the layout algorithm for graph visualization. Note: hierarchical layout works best with directed acyclic graphs"
            )
            
            # Node size metric
            size_metric = st.selectbox(
                "Node Size Based On",
                ["degree_centrality", "betweenness_centrality", "pagerank", "degree", "risk_score"],
                help="Choose the metric to determine node sizes"
            )
            
            # Analysis level
            analysis_level = st.selectbox(
                "Analysis Level",
                ["Level 1 (Basic)", "Level 2 (Advanced)"],
                help="Choose the depth of analysis"
            )
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary"):
                run_analysis(analysis_level)
    
    # Main content area with tabs
    if st.session_state.data_loaded:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🕸️ Graph Visualization", "📈 Risk Metrics", "🔍 Advanced Analysis", "🛡️ Scenario Planning"])
        
        with tab1:
            show_data_overview()
        
        with tab2:
            show_graph_visualization(layout_option, size_metric)
        
        with tab3:
            show_risk_metrics()
        
        with tab4:
            show_advanced_analysis()
        
        with tab5:
            show_scenario_planning()
    
    elif st.session_state.show_mapping:
        # Show column mapping interface
        show_smart_column_mapping()
    
    else:
        # Welcome screen
        show_welcome_screen()

def run_analysis(analysis_level):
    """Run the supply chain analysis"""
    with st.spinner("🔄 Running analysis..."):
        try:
            # Validate and build graph
            is_valid, message = st.session_state.data_processor.validate_data()
            if not is_valid:
                st.error(f"Data validation failed: {message}")
                return
            
            st.success(message)
            
            # Build graph
            if not st.session_state.data_processor.build_graph():
                st.error("Failed to build graph from data")
                return
            
            # Initialize analyzer
            st.session_state.analyzer = GraphAnalyzer(
                st.session_state.data_processor.graph,
                st.session_state.data_processor.node_types
            )
            
            # Calculate metrics
            st.session_state.analyzer.calculate_basic_metrics()
            
            if analysis_level == "Level 2 (Advanced)":
                # Advanced analysis
                st.session_state.analyzer.calculate_risk_scores()
                st.session_state.analyzer.detect_communities()
                st.session_state.analyzer.detect_anomalies()
                
                # Initialize advanced analyzer
                st.session_state.advanced_analyzer = AdvancedRiskAnalyzer(
                    st.session_state.data_processor.graph,
                    st.session_state.data_processor.node_types,
                    st.session_state.analyzer.metrics
                )
                st.session_state.advanced_analyzer.calculate_advanced_risk_scores()
            
            # Initialize visualizer
            st.session_state.visualizer = SupplyChainVisualizer(
                st.session_state.data_processor.graph,
                st.session_state.data_processor.node_types,
                st.session_state.analyzer.metrics,
                st.session_state.analyzer.risk_scores
            )
            
            st.session_state.analysis_complete = True
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def show_smart_column_mapping():
    """Show intelligent column mapping interface for any dataset"""
    st.header("🔄 Smart Data Mapping")
    
    st.info("""
    🤖 **Smart Detection**: ChainSense will help you create supply chain relationships from your data.
    Choose how you want to model your supply chain network.
    """)
    
    # Show data preview
    st.subheader("👀 Your Data Preview")
    st.dataframe(st.session_state.data_processor.df.head(10), use_container_width=True)
    
    # Available columns
    available_cols = list(st.session_state.data_processor.df.columns)
    
    st.subheader("🎯 Choose Your Mapping Strategy")
    
    strategy = st.radio(
        "How would you like to create supply chain relationships?",
        [
            "🏭 Supplier → Geographic Markets",
            "📦 Supplier → Product Categories", 
            "🔄 Custom Two-Column Mapping",
            "🎯 Smart Auto-Detection"
        ]
    )
    
    if strategy == "🏭 Supplier → Geographic Markets":
        col1, col2 = st.columns(2)
        
        with col1:
            supplier_col = st.selectbox(
                "Select Supplier Column:",
                available_cols,
                index=next((i for i, col in enumerate(available_cols) if 'supplier' in col.lower()), 0)
            )
        
        with col2:
            location_cols = [col for col in available_cols if any(keyword in col.lower() for keyword in ['location', 'region', 'city', 'country', 'area'])]
            location_col = st.selectbox(
                "Select Location/Market Column:",
                location_cols if location_cols else available_cols,
                index=0
            )
        
        if st.button("⚙️ Create Geographic Relationships", type="primary"):
            create_geographic_relationships(supplier_col, location_col)
    
    elif strategy == "📦 Supplier → Product Categories":
        col1, col2 = st.columns(2)
        
        with col1:
            supplier_col = st.selectbox(
                "Select Supplier Column:",
                available_cols,
                index=next((i for i, col in enumerate(available_cols) if 'supplier' in col.lower()), 0)
            )
        
        with col2:
            product_cols = [col for col in available_cols if any(keyword in col.lower() for keyword in ['product', 'category', 'type', 'item'])]
            product_col = st.selectbox(
                "Select Product/Category Column:",
                product_cols if product_cols else available_cols,
                index=0
            )
        
        if st.button("⚙️ Create Product Relationships", type="primary"):
            create_product_relationships(supplier_col, product_col)
    
    elif strategy == "🔄 Custom Two-Column Mapping":
        col1, col2 = st.columns(2)
        
        with col1:
            source_col = st.selectbox(
                "Select Source/Supplier Column:",
                available_cols
            )
        
        with col2:
            target_col = st.selectbox(
                "Select Target/Customer Column:",
                [col for col in available_cols if col != source_col]
            )
        
        if st.button("⚙️ Create Custom Relationships", type="primary"):
            create_custom_relationships(source_col, target_col)
    
    elif strategy == "🎯 Smart Auto-Detection":
        st.info("""
        🤖 **Auto-Detection** will analyze your data and create the most logical supply chain relationships 
        based on data patterns, column names, and value distributions.
        """)
        
        if st.button("🔍 Run Smart Auto-Detection", type="primary"):
            run_smart_auto_detection()
    
    # Cancel option
    if st.button("❌ Cancel and Upload Different File"):
        st.session_state.show_mapping = False
        st.session_state.data_processor.df = None
        st.rerun()

def create_geographic_relationships(supplier_col: str, location_col: str):
    """Create supplier to geographic market relationships"""
    try:
        relationships = []
        for _, row in st.session_state.data_processor.df.iterrows():
            supplier = str(row.get(supplier_col, '')).strip()
            location = str(row.get(location_col, '')).strip()
            
            if supplier and location and supplier.lower() not in ['nan', '', 'null'] and location.lower() not in ['nan', '', 'null']:
                relationships.append({
                    'supplier': supplier,
                    'customer': f"Market_{location}"
                })
        
        if relationships:
            new_df = pd.DataFrame(relationships).drop_duplicates()
            st.session_state.data_processor.df = new_df
            st.session_state.data_loaded = True
            st.session_state.show_mapping = False
            st.success(f"✨ Created {len(new_df)} geographic relationships successfully!")
            st.rerun()
        else:
            st.error("No valid relationships could be created. Please check your data.")
    except Exception as e:
        st.error(f"Error creating relationships: {str(e)}")

def create_product_relationships(supplier_col: str, product_col: str):
    """Create supplier to product category relationships"""
    try:
        relationships = []
        for _, row in st.session_state.data_processor.df.iterrows():
            supplier = str(row.get(supplier_col, '')).strip()
            product = str(row.get(product_col, '')).strip()
            
            if supplier and product and supplier.lower() not in ['nan', '', 'null'] and product.lower() not in ['nan', '', 'null']:
                relationships.append({
                    'supplier': supplier,
                    'customer': f"ProductLine_{product}"
                })
        
        if relationships:
            new_df = pd.DataFrame(relationships).drop_duplicates()
            st.session_state.data_processor.df = new_df
            st.session_state.data_loaded = True
            st.session_state.show_mapping = False
            st.success(f"✨ Created {len(new_df)} product relationships successfully!")
            st.rerun()
        else:
            st.error("No valid relationships could be created. Please check your data.")
    except Exception as e:
        st.error(f"Error creating relationships: {str(e)}")

def create_custom_relationships(source_col: str, target_col: str):
    """Create custom two-column relationships"""
    try:
        relationships = []
        for _, row in st.session_state.data_processor.df.iterrows():
            source = str(row.get(source_col, '')).strip()
            target = str(row.get(target_col, '')).strip()
            
            if (source and target and 
                source.lower() not in ['nan', '', 'null'] and 
                target.lower() not in ['nan', '', 'null'] and 
                source != target):
                relationships.append({
                    'supplier': source,
                    'customer': target
                })
        
        if relationships:
            new_df = pd.DataFrame(relationships).drop_duplicates()
            st.session_state.data_processor.df = new_df
            st.session_state.data_loaded = True
            st.session_state.show_mapping = False
            st.success(f"✨ Created {len(new_df)} custom relationships successfully!")
            st.rerun()
        else:
            st.error("No valid relationships could be created. Please check your data and ensure the columns have different values.")
    except Exception as e:
        st.error(f"Error creating relationships: {str(e)}")

def run_smart_auto_detection():
    """Run intelligent auto-detection"""
    with st.spinner("🔍 Running smart analysis..."):
        if st.session_state.data_processor.auto_detect_relationships():
            st.session_state.data_loaded = True
            st.session_state.show_mapping = False
            st.success("✨ Smart auto-detection created supply chain relationships successfully!")
            st.rerun()
        else:
            st.error("🙁 Auto-detection could not find suitable relationships. Please try manual mapping.")

def show_welcome_screen():
    """Show welcome screen with instructions"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Welcome to ChainSense!
        
        **ChainSense** is an AI-powered supply chain risk analyzer that helps you:
        
        ### 📊 Level 1 - Exploratory Dashboard
        - Upload and analyze supply chain datasets
        - Visualize supply chain networks interactively
        - Calculate basic graph metrics (centrality, clustering)
        - Identify critical nodes and bottlenecks
        
        ### 🎯 Level 2 - Advanced Risk Analysis
        - Calculate risk scores for suppliers and customers
        - Detect communities and clusters in your network
        - Identify anomalous nodes and relationships
        - Generate comprehensive risk assessments
        
        ### 🚀 Getting Started
        
        1. **Upload Data**: Use the sidebar to upload a CSV file with your supply chain data
        2. **Required Columns**: `supplier` and `customer` (case-insensitive)
        3. **Optional Columns**: `product`, `quantity`, `price`, `date`
        4. **Or Try Sample Data**: Click "Load Sample Data" to explore with example data
        
        ### 📋 Data Format Example
        ```
        supplier,customer,product,quantity
        Supplier_A,Distributor_X,Product_Alpha,500
        Distributor_X,Retailer_1,Product_Alpha,100
        ```
        """)
        
        # Add feature highlights
        st.markdown("""
        ### ✨ Key Features
        """)
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.info("""
            **🔍 Graph Analytics**
            - Node centrality analysis
            - Bottleneck identification
            - Network connectivity analysis
            - Community detection
            """)
        
        with feature_col2:
            st.info("""
            **⚠️ Risk Assessment**
            - Automated risk scoring
            - Anomaly detection
            - Vulnerability analysis
            - Interactive visualizations
            """)

def show_data_overview():
    """Show data overview tab"""
    if st.session_state.data_processor.df is None:
        st.warning("No data loaded. Please upload a CSV file or load sample data.")
        return
    
    st.header("📊 Data Overview")
    
    # Data summary with enhanced styling
    data_summary = st.session_state.data_processor.get_data_summary()
    
    # Handle case where data_summary might be None or empty
    if not data_summary:
        data_summary = {
            'total_rows': 0,
            'unique_suppliers': 0,
            'unique_customers': 0,
            'columns': []
        }
    
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h3 style="color: #1f77b4; margin-bottom: 1rem;">📊 Key Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">{}</div>
            <div class="stats-label">Total Records</div>
        </div>
        """.format(data_summary.get('total_rows', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">{}</div>
            <div class="stats-label">Unique Suppliers</div>
        </div>
        """.format(data_summary.get('unique_suppliers', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">{}</div>
            <div class="stats-label">Unique Customers</div>
        </div>
        """.format(data_summary.get('unique_customers', 0)), unsafe_allow_html=True)
    
    with col4:
        total_entities = data_summary.get('unique_suppliers', 0) + data_summary.get('unique_customers', 0)
        st.markdown("""
        <div class="stats-card">
            <div class="stats-number">{}</div>
            <div class="stats-label">Total Entities</div>
        </div>
        """.format(total_entities), unsafe_allow_html=True)
    
    # Display data table
    st.subheader("📋 Raw Data")
    
    # Data filtering options
    col1, col2 = st.columns(2)
    
    # Initialize filter variables
    suppliers = []
    customers = []
    
    with col1:
        if 'supplier' in st.session_state.data_processor.df.columns:
            suppliers = st.multiselect(
                "Filter by Supplier",
                st.session_state.data_processor.df['supplier'].unique()
            )
    
    with col2:
        if 'customer' in st.session_state.data_processor.df.columns:
            customers = st.multiselect(
                "Filter by Customer",
                st.session_state.data_processor.df['customer'].unique()
            )
    
    # Apply filters
    filtered_df = st.session_state.data_processor.df.copy()
    
    if suppliers:
        filtered_df = filtered_df[filtered_df['supplier'].isin(suppliers)]
    
    if customers:
        filtered_df = filtered_df[filtered_df['customer'].isin(customers)]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Data statistics
    if st.session_state.analysis_complete:
        st.subheader("📈 Graph Statistics")
        graph_summary = st.session_state.data_processor.get_graph_summary()
        
        # Handle case where graph_summary might be None or empty
        if not graph_summary:
            graph_summary = {
                'total_nodes': 0,
                'total_edges': 0,
                'density': 0.0,
                'connected_components': 0,
                'is_connected': False,
                'node_types': {}
            }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", graph_summary.get('total_nodes', 0))
            st.metric("Total Edges", graph_summary.get('total_edges', 0))
        
        with col2:
            st.metric("Network Density", f"{graph_summary.get('density', 0):.3f}")
            st.metric("Connected Components", graph_summary.get('connected_components', 0))
        
        with col3:
            is_connected = graph_summary.get('is_connected', False)
            st.metric("Is Connected", "Yes" if is_connected else "No")
        
        # Node type distribution
        if 'node_types' in graph_summary:
            node_types = graph_summary['node_types']
            
            st.subheader("🏷️ Node Type Distribution")
            
            # Create pie chart
            labels = []
            values = []
            colors = []
            
            color_map = {
                'supplier': '#FF6B6B',
                'customer': '#4ECDC4',
                'intermediary': '#45B7D1',
                'isolated': '#96CEB4'
            }
            
            for node_type, count in node_types.items():
                if count > 0:
                    labels.append(node_type.title())
                    values.append(count)
                    colors.append(color_map.get(node_type, '#CCCCCC'))
            
            if values:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors,
                    hole=0.4
                )])
                
                fig_pie.update_layout(
                    title="Node Type Distribution",
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)

def show_graph_visualization(layout_option, size_metric):
    """Show graph visualization tab"""
    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first to see graph visualization.")
        return
    
    st.header("🕸️ Graph Visualization")
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_risk_colors = st.checkbox("Show Risk Colors", value=False)
    
    with col2:
        graph_height = st.selectbox("Graph Height", ["400px", "600px", "800px"], index=1)
    
    with col3:
        show_labels = st.checkbox("Show Node Labels", value=True)
    
    # Create interactive graph
    try:
        graph_file = st.session_state.visualizer.create_interactive_graph(
            layout=layout_option,
            node_size_metric=size_metric,
            show_risk=show_risk_colors,
            height=graph_height
        )
        
        if graph_file:
            # Display the graph
            with open(graph_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=int(graph_height.replace('px', '')) + 50)
            
            # Clean up temporary file
            os.unlink(graph_file)
        
    except Exception as e:
        st.error(f"Error creating graph visualization: {str(e)}")
    
    # Network overview using plotly
    st.subheader("📊 Network Overview")
    
    try:
        overview_fig = st.session_state.visualizer.create_network_overview()
        if overview_fig:
            st.plotly_chart(overview_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating network overview: {str(e)}")

def show_risk_metrics():
    """Show risk metrics tab"""
    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first to see risk metrics.")
        return
    
    st.header("📈 Risk Metrics")
    
    # Basic metrics
    metrics = st.session_state.analyzer.metrics
    
    if metrics:
        st.subheader("🎯 Centrality Analysis")
        
        # Top nodes by different centralities
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔝 Top Nodes by Degree Centrality**")
            top_degree = st.session_state.analyzer.get_top_nodes('degree_centrality', 10)
            for i, (node, score) in enumerate(top_degree, 1):
                node_type = st.session_state.data_processor.node_types.get(node, 'unknown')
                st.write(f"{i}. **{node}** ({node_type}): {score:.3f}")
        
        with col2:
            st.markdown("**🔄 Top Nodes by Betweenness Centrality**")
            top_betweenness = st.session_state.analyzer.get_top_nodes('betweenness_centrality', 10)
            for i, (node, score) in enumerate(top_betweenness, 1):
                node_type = st.session_state.data_processor.node_types.get(node, 'unknown')
                st.write(f"{i}. **{node}** ({node_type}): {score:.3f}")
    
    # Risk scores (Level 2)
    if st.session_state.analyzer.risk_scores:
        st.subheader("⚠️ Risk Assessment")
        
        # Risk statistics
        risk_values = list(st.session_state.analyzer.risk_scores.values())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Risk", f"{np.mean(risk_values):.3f}")
        
        with col2:
            high_risk_count = sum(1 for r in risk_values if r > 0.7)
            st.metric("High Risk Nodes", high_risk_count)
        
        with col3:
            medium_risk_count = sum(1 for r in risk_values if 0.3 <= r <= 0.7)
            st.metric("Medium Risk Nodes", medium_risk_count)
        
        with col4:
            low_risk_count = sum(1 for r in risk_values if r < 0.3)
            st.metric("Low Risk Nodes", low_risk_count)
        
        # Risk heatmap
        risk_heatmap = st.session_state.visualizer.create_risk_heatmap()
        if risk_heatmap:
            st.plotly_chart(risk_heatmap, use_container_width=True)
        
        # Advanced risk dashboard (Level 2)
        if st.session_state.advanced_analyzer and hasattr(st.session_state.advanced_analyzer, 'risk_factors'):
            if st.session_state.advanced_analyzer.risk_factors:
                st.subheader("🔬 Advanced Risk Analysis")
                
                advanced_figures = st.session_state.visualizer.create_advanced_risk_dashboard(
                    st.session_state.advanced_analyzer.risk_factors
                )
                
                for fig in advanced_figures:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Top risky nodes
        st.subheader("🚨 Top Risky Nodes")
        
        sorted_risks = sorted(st.session_state.analyzer.risk_scores.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for i, (node, risk_score) in enumerate(sorted_risks[:10], 1):
            node_type = st.session_state.data_processor.node_types.get(node, 'unknown')
            
            if risk_score > 0.7:
                risk_class = "risk-high"
                risk_level = "HIGH"
            elif risk_score > 0.3:
                risk_class = "risk-medium"
                risk_level = "MEDIUM"
            else:
                risk_class = "risk-low"
                risk_level = "LOW"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <strong>{i}. {node}</strong> ({node_type})<br>
                Risk Score: {risk_score:.3f} - <strong>{risk_level} RISK</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Vulnerability analysis
    vulnerability = st.session_state.analyzer.get_vulnerability_analysis()
    if vulnerability:
        st.subheader("🛡️ Vulnerability Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Vulnerability Score", f"{vulnerability.get('vulnerability_score', 0):.3f}")
            st.metric("Articulation Points", len(vulnerability.get('articulation_points', [])))
        
        with col2:
            st.metric("Critical Bridges", len(vulnerability.get('bridges', [])))
            st.metric("Bottlenecks", len(vulnerability.get('bottlenecks', [])))
        
        # Show critical nodes
        if vulnerability.get('articulation_points'):
            st.markdown("**🔴 Critical Nodes (Single Points of Failure):**")
            for point in vulnerability['articulation_points'][:5]:
                node_type = st.session_state.data_processor.node_types.get(point, 'unknown')
                st.write(f"• **{point}** ({node_type})")

def show_advanced_analysis():
    """Show advanced analysis tab (Level 2 features)"""
    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first to see advanced features.")
        return
    
    st.header("🔍 Advanced Analysis")
    
    # Community detection
    if st.session_state.analyzer.communities:
        st.subheader("🏘️ Community Detection")
        
        communities = st.session_state.analyzer.communities
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Communities", communities.get('num_communities', 0))
        
        with col2:
            # Average community size
            if 'stats' in communities:
                sizes = [stats['size'] for stats in communities['stats'].values()]
                avg_size = np.mean(sizes) if sizes else 0
                st.metric("Average Community Size", f"{avg_size:.1f}")
        
        # Community visualization
        community_fig = st.session_state.visualizer.create_community_visualization(communities)
        if community_fig:
            st.plotly_chart(community_fig, use_container_width=True)
        
        # Community details
        st.subheader("📋 Community Details")
        
        if 'stats' in communities:
            for comm_id, stats in communities['stats'].items():
                with st.expander(f"Community {comm_id} ({stats['size']} nodes)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Nodes:**")
                        for node in stats['nodes'][:10]:  # Show first 10 nodes
                            node_type = st.session_state.data_processor.node_types.get(node, 'unknown')
                            st.write(f"• {node} ({node_type})")
                        
                        if len(stats['nodes']) > 10:
                            st.write(f"... and {len(stats['nodes']) - 10} more")
                    
                    with col2:
                        st.write("**Node Type Distribution:**")
                        for node_type, count in stats['node_types'].items():
                            if count > 0:
                                st.write(f"• {node_type.title()}: {count}")
                        
                        st.metric("Community Density", f"{stats['density']:.3f}")
    
    # Anomaly detection
    if st.session_state.analyzer.anomalies:
        st.subheader("🚨 Anomaly Detection")
        
        anomalies = st.session_state.analyzer.anomalies
        anomalous_nodes = [node for node, data in anomalies.items() if data['is_anomaly']]
        
        st.metric("Anomalous Nodes Found", len(anomalous_nodes))
        
        if anomalous_nodes:
            st.markdown("**🔍 Detected Anomalies:**")
            
            # Sort by anomaly score
            sorted_anomalies = sorted(
                [(node, data) for node, data in anomalies.items() if data['is_anomaly']],
                key=lambda x: x[1]['anomaly_score'],
                reverse=True
            )
            
            for node, data in sorted_anomalies[:10]:
                node_type = st.session_state.data_processor.node_types.get(node, 'unknown')
                score = data['anomaly_score']
                method = data['method']
                
                st.markdown(f"""
                <div class="risk-high">
                    <strong>{node}</strong> ({node_type})<br>
                    Anomaly Score: {score:.3f} - Method: {method}
                </div>
                """, unsafe_allow_html=True)
    
    # Supply chain insights
    insights = st.session_state.analyzer.get_supply_chain_insights()
    if insights:
        st.subheader("💡 Supply Chain Insights")
        
        if 'structure' in insights:
            structure = insights['structure']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Supplier Ratio", f"{structure.get('supplier_ratio', 0):.1%}")
            
            with col2:
                st.metric("Customer Ratio", f"{structure.get('customer_ratio', 0):.1%}")
            
            with col3:
                st.metric("Supply Chain Depth", f"{structure.get('supply_chain_depth', 0)} tiers")
        
        if 'connectivity' in insights:
            connectivity = insights['connectivity']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Degree", f"{connectivity.get('mean_degree', 0):.1f}")
                st.metric("Isolated Nodes", connectivity.get('isolated_nodes', 0))
            
            with col2:
                st.metric("Highly Connected Nodes", connectivity.get('highly_connected_nodes', 0))
                st.metric("Degree Variance", f"{connectivity.get('degree_variance', 0):.2f}")

def show_scenario_planning():
    """Show scenario planning and simulation tab"""
    if not st.session_state.analysis_complete or st.session_state.advanced_analyzer is None:
        st.warning("Please run Level 2 analysis first to access scenario planning.")
        return
    
    st.header("🛡️ Scenario Planning & Simulation")
    
    # Scenario configuration
    st.subheader("🎯 Configure Disruption Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔥 Predefined Scenarios**")
        
        if st.button("🏭 Simulate Major Supplier Disruption"):
            # Find top suppliers by out-degree
            suppliers = [node for node in st.session_state.data_processor.graph.nodes() 
                        if st.session_state.data_processor.node_types.get(node) == 'supplier']
            
            if suppliers:
                # Get top 2 suppliers by connections
                supplier_degrees = [(node, st.session_state.data_processor.graph.out_degree(node)) 
                                  for node in suppliers]
                supplier_degrees.sort(key=lambda x: x[1], reverse=True)
                top_suppliers = [node for node, _ in supplier_degrees[:2]]
                
                scenarios = [{
                    'name': 'Major Supplier Disruption',
                    'nodes': top_suppliers,
                    'type': 'removal'
                }]
                
                results = st.session_state.advanced_analyzer.simulate_disruption_scenarios(scenarios)
                display_scenario_results(results)
        
        if st.button("🚪 Simulate Key Intermediary Failure"):
            # Find key intermediaries
            intermediaries = [node for node in st.session_state.data_processor.graph.nodes() 
                            if st.session_state.data_processor.node_types.get(node) == 'intermediary']
            
            if intermediaries and 'betweenness_centrality' in st.session_state.analyzer.metrics:
                # Get highest betweenness centrality intermediary
                intermediary_centrality = [(node, st.session_state.analyzer.metrics['betweenness_centrality'].get(node, 0)) 
                                         for node in intermediaries]
                intermediary_centrality.sort(key=lambda x: x[1], reverse=True)
                
                if intermediary_centrality:
                    key_intermediary = intermediary_centrality[0][0]
                    
                    scenarios = [{
                        'name': 'Key Intermediary Failure',
                        'nodes': [key_intermediary],
                        'type': 'removal'
                    }]
                    
                    results = st.session_state.advanced_analyzer.simulate_disruption_scenarios(scenarios)
                    display_scenario_results(results)
        
        if st.button("🌊 Simulate Regional Disruption"):
            # Simulate community-based disruption
            if st.session_state.analyzer.communities:
                communities = st.session_state.analyzer.communities['assignments']
                community_sizes = {}
                
                for node, comm_id in communities.items():
                    if comm_id not in community_sizes:
                        community_sizes[comm_id] = []
                    community_sizes[comm_id].append(node)
                
                # Find largest community
                largest_comm = max(community_sizes.items(), key=lambda x: len(x[1]))
                
                scenarios = [{
                    'name': 'Regional Disruption',
                    'nodes': largest_comm[1][:5],  # Disrupt top 5 nodes in largest community
                    'type': 'removal'
                }]
                
                results = st.session_state.advanced_analyzer.simulate_disruption_scenarios(scenarios)
                display_scenario_results(results)
    
    with col2:
        st.markdown("**🎲 Custom Scenario Builder**")
        
        # Custom scenario builder
        scenario_name = st.text_input("Scenario Name", value="Custom Scenario")
        
        # Node selection
        all_nodes = list(st.session_state.data_processor.graph.nodes())
        selected_nodes = st.multiselect(
            "Select Nodes to Disrupt",
            all_nodes,
            help="Choose nodes that will be affected in this scenario"
        )
        
        disruption_type = st.selectbox(
            "Disruption Type",
            ["removal", "capacity_reduction"],
            help="removal: Complete node failure, capacity_reduction: Partial capacity loss"
        )
        
        if st.button("🚀 Run Custom Scenario") and selected_nodes:
            scenarios = [{
                'name': scenario_name,
                'nodes': selected_nodes,
                'type': disruption_type
            }]
            
            results = st.session_state.advanced_analyzer.simulate_disruption_scenarios(scenarios)
            display_scenario_results(results)
    
    # Critical paths analysis
    st.subheader("🛜 Critical Paths Analysis")
    
    if st.button("🔍 Identify Critical Paths"):
        critical_paths = st.session_state.advanced_analyzer.identify_critical_paths()
        
        if critical_paths:
            st.markdown("**🎆 Most Critical Supply Chain Paths:**")
            
            for i, path_info in enumerate(critical_paths[:5], 1):
                path = path_info['path']
                criticality = path_info['criticality']
                length = path_info['length']
                
                path_str = " → ".join(path)
                
                with st.expander(f"Path {i}: {path_info['supplier']} to {path_info['customer']} (Criticality: {criticality:.3f})"):
                    st.write(f"**Full Path:** {path_str}")
                    st.write(f"**Path Length:** {length} hops")
                    st.write(f"**Criticality Score:** {criticality:.3f}")
                    
                    # Analyze each node in the path
                    st.write("**Node Analysis:**")
                    for node in path:
                        node_type = st.session_state.data_processor.node_types.get(node, 'unknown')
                        degree = st.session_state.data_processor.graph.degree(node)
                        
                        risk_info = ""
                        if hasattr(st.session_state.advanced_analyzer, 'risk_factors') and st.session_state.advanced_analyzer.risk_factors:
                            if 'composite' in st.session_state.advanced_analyzer.risk_factors:
                                risk_score = st.session_state.advanced_analyzer.risk_factors['composite'].get(node, 0)
                                risk_info = f" (Risk: {risk_score:.3f})"
                        
                        st.write(f"  • **{node}** ({node_type}, degree: {degree}){risk_info}")
        else:
            st.info("No critical paths found. This might indicate a very sparse or disconnected network.")
    
    # Resilience recommendations
    st.subheader("📝 Resilience Recommendations")
    
    if st.button("💡 Generate Recommendations"):
        recommendations = st.session_state.advanced_analyzer.generate_resilience_recommendations()
        
        if recommendations:
            # Group recommendations by type
            rec_types = {}
            for rec in recommendations:
                rec_type = rec['type']
                if rec_type not in rec_types:
                    rec_types[rec_type] = []
                rec_types[rec_type].append(rec)
            
            for rec_type, recs in rec_types.items():
                st.markdown(f"**🎯 {rec_type.title()} Recommendations:**")
                
                for rec in recs:
                    priority = rec.get('priority', 'medium')
                    priority_color = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(priority, '🟡')
                    
                    st.markdown(f"{priority_color} **{rec['description']}**")
                    st.markdown(f"   *Impact:* {rec['impact']}")
                    
                    if 'node' in rec:
                        st.markdown(f"   *Target:* {rec['node']} ({rec.get('node_type', 'unknown')})")
                    
                    st.markdown("---")
        else:
            st.info("No specific recommendations generated. Your supply chain appears to be well-structured.")

def display_scenario_results(results: Dict):
    """Display simulation results"""
    if not results:
        st.error("No simulation results to display.")
        return
    
    st.subheader("📈 Simulation Results")
    
    for scenario_name, result in results.items():
        st.markdown(f"**🎯 Scenario: {scenario_name}**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes Removed", result.get('nodes_removed', 0))
        
        with col2:
            st.metric("Edges Lost", result.get('edges_lost', 0))
        
        with col3:
            connectivity_loss = result.get('connectivity_loss', 0)
            st.metric("Connectivity Loss", f"{connectivity_loss:.1%}")
        
        with col4:
            criticality = result.get('criticality_score', 0)
            st.metric("Criticality Score", f"{criticality:.3f}")
        
        # Impact assessment
        connectivity_loss = result.get('connectivity_loss', 0)
        fragmentation = result.get('fragmentation', 0)
        isolation_impact = result.get('isolation_impact', 0)
        
        if connectivity_loss > 0.5:
            impact_level = "🔴 SEVERE"
            impact_color = "risk-high"
        elif connectivity_loss > 0.3:
            impact_level = "🟠 HIGH"
            impact_color = "risk-medium"
        elif connectivity_loss > 0.1:
            impact_level = "🟡 MEDIUM"
            impact_color = "risk-medium"
        else:
            impact_level = "🟢 LOW"
            impact_color = "risk-low"
        
        st.markdown(f"""
        <div class="{impact_color}">
            <strong>Impact Assessment: {impact_level}</strong><br>
            Network Fragmentation: {fragmentation} new components<br>
            Isolation Impact: {isolation_impact:.1%} of nodes isolated<br>
            Largest Remaining Component: {result.get('largest_component_size', 0)} nodes
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

if __name__ == "__main__":
    main()