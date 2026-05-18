import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import json
import random
import numpy as np
import requests
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ChainSense | Pro Supply Chain Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
BACKEND_URL = "https://ehtel-deserted-boris.ngrok-free.dev" # Ganti dengan URL Ngrok dari Kaggle

TARGET_SCHEMA = {
    'Vendor_Name': 'Name of the supplier/vendor',
    'Customer_Location': 'Destination city/location',
    'Order_Date': 'Date of order/transaction',
    'Delivery_Status': 'Status (Late, On Time, etc)',
    'Risk_Score': 'Numerical risk value (0-100)',
    'Order_ID': 'Unique identifier for the order',
    'Quantity': 'Number of items',
    'Shipping_Cost': 'Cost of shipping',
    'Actual_Shipping_Days': 'Number of days taken to ship',
    'Product_Category': 'Category of product'
}

# Custom CSS for Professional Dark UI
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #4da6ff;
    }
    div[data-testid="stMetric"] > div {
        color: #fafafa !important;
    }
    p[data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #fafafa !important;
    }
    
    /* Custom Container */
    .stContainer {
        background-color: #262730;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #303030;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: #262730;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 5px;
        color: #b0b0b0;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #303030;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0e1117;
        color: #4da6ff;
        border-bottom: 2px solid #4da6ff;
    }
    
    /* Multiselect & Inputs */
    .stMultiSelect > div > div > div {
        background-color: #262730;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_data(file_input=None):
    try:
        if file_input is not None:
            df = pd.read_csv(file_input)
        else:
            # Load default synthetic data
            df = pd.read_csv("chainsense_synthetic_data.csv")
        
        # Ensure column types
        if 'Order_Date' in df.columns:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def validate_data(df):
    """Check if uploaded data has required columns"""
    required_columns = [
        'Vendor_Name', 'Delivery_Status', 'Risk_Score', 
        'Order_ID', 'Customer_Location', 'Quantity', 'Shipping_Cost', 
        'Actual_Shipping_Days'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    return True, "Data is valid"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_kpis(df):
    total_orders = len(df)
    
    # On-Time Performance
    on_time_orders = len(df[df['Delivery_Status'] == 'On Time'])
    on_time_pct = (on_time_orders / total_orders) * 100 if total_orders > 0 else 0
    
    # Critical Vendors (Risk Score > 75)
    high_risk_vendors = df[df['Risk_Score'] > 75]['Vendor_Name'].nunique()
    
    # Avg Shipping Cost
    avg_cost = df['Shipping_Cost'].mean()
    
    return total_orders, on_time_pct, high_risk_vendors, avg_cost

def get_gemma_analysis(summary_text):
    """Call Kaggle Backend API for intelligence briefing"""
    try:
        payload = {
            "task": "analysis",
            "summary": summary_text
        }
        headers = {"ngrok-skip-browser-warning": "true"}
        
        response = requests.post(f"{BACKEND_URL}/analyze", json=payload, headers=headers, timeout=180)
        
        if response.status_code == 200:
            return response.json().get('response', "⚠️ No response content.")
        else:
            return f"⚠️ Backend Error: Status {response.status_code}. Check your Kaggle terminal."
            
    except requests.exceptions.ConnectionError:
        return "❌ Failed to connect to Kaggle Backend. Ensure Ngrok tunnel is active and BACKEND_URL is correct in app.py."
    except Exception as e:
        return f"❌ AI Analysis Failed: {str(e)}"

def get_gnn_prediction(sim_df):
    """Prepare graph data and call Kaggle Backend API for GNN risk prediction"""
    try:
        # Build graph
        G = nx.from_pandas_edgelist(sim_df, source='Vendor_Name', target='Customer_Location', edge_attr='Quantity')
        
        degree_c = nx.degree_centrality(G)
        between_c = nx.betweenness_centrality(G)
        
        # Build node index map
        node_names = list(G.nodes())
        node_to_idx = {name: i for i, name in enumerate(node_names)}
        
        # Edge list
        edge_list = [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges()]
        
        # Helper dictionaries for fast lookup
        vendor_stats = sim_df.groupby('Vendor_Name').agg({
            'New_Risk_Score': 'mean',
            'Actual_Shipping_Days': 'mean',
            'Delivery_Status': lambda x: (x == 'On Time').mean() if len(x) > 0 else 0
        }).to_dict('index')
        
        max_risk = sim_df['New_Risk_Score'].max() or 1
        max_ship = sim_df['Actual_Shipping_Days'].max() or 1
        
        node_features = []
        for node in node_names:
            dc = degree_c.get(node, 0)
            bc = between_c.get(node, 0)
            
            if node in vendor_stats:
                st = vendor_stats[node]
                risk_norm = st['New_Risk_Score'] / max_risk
                ship_norm = st['Actual_Shipping_Days'] / max_ship
                ontime_ratio = st['Delivery_Status']
            else:
                risk_norm = 0.0
                ship_norm = 0.0
                ontime_ratio = 1.0
                
            node_features.append([dc, bc, risk_norm, ship_norm, ontime_ratio])
            
        payload = {
            "task": "gnn_predict",
            "node_features": node_features,
            "edge_list": edge_list,
            "node_names": node_names
        }
        headers = {"ngrok-skip-browser-warning": "true"}
        
        response = requests.post(f"{BACKEND_URL}/analyze", json=payload, headers=headers, timeout=180)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Backend Error: {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Failed to connect to Backend. Ensure Ngrok is active."}
    except Exception as e:
        return {"error": f"❌ GNN Analysis Failed: {str(e)}"}

def smart_map_columns(df):
    """
    Use Kaggle Backend AI to map user columns to ChainSense standard schema.
    """
    try:
        user_columns = list(df.columns)
        payload = {
            "task": "mapping",
            "target_schema": TARGET_SCHEMA,
            "user_columns": user_columns
        }
        headers = {"ngrok-skip-browser-warning": "true"}
        
        response = requests.post(f"{BACKEND_URL}/analyze", json=payload, headers=headers, timeout=180)
        
        if response.status_code == 200:
            mapping = response.json().get('mapping', {})
            return mapping
        else:
            st.error(f"Backend Error: Status {response.status_code}")
            return {}
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Failed to connect to Kaggle Backend. Check your Ngrok connection.")
        return {}
    except Exception as e:
        st.error(f"AI Mapping Failed: {str(e)}")
        return {}

def ensure_critical_columns(df):
    """
    Polyfill missing critical columns with synthetic data if possible.
    Returns: df, was_modified, modifications_list
    """
    modified = False
    modifications = []
    
    # 1. Actual_Shipping_Days
    if 'Actual_Shipping_Days' not in df.columns:
        if 'Delivery_Status' in df.columns:
            # Generate based on status
            def est_days(status):
                if status == 'Late': return random.randint(7, 15)
                if status == 'On Time': return random.randint(2, 6)
                return random.randint(3, 10)
            
            df['Actual_Shipping_Days'] = df['Delivery_Status'].apply(est_days)
            modifications.append("Generated 'Actual_Shipping_Days' based on 'Delivery_Status'")
            modified = True
        else:
            # Random default
            df['Actual_Shipping_Days'] = np.random.randint(1, 10, size=len(df))
            modifications.append("Generated random 'Actual_Shipping_Days'")
            modified = True

    # 2. Risk_Score
    if 'Risk_Score' not in df.columns:
        df['Risk_Score'] = np.random.randint(0, 50, size=len(df))
        modifications.append("Generated random 'Risk_Score'")
        modified = True
        
    return df, modified, modifications

# ==========================================
# MAIN APP
# ==========================================
def main():
    # Sidebar for Data Input
    with st.sidebar:
        st.header("📂 Data Config")
        
        # Template Download
        try:
            with open("chainsense_synthetic_data.csv", "rb") as f:
                st.download_button(
                    label="📥 Download Template",
                    data=f,
                    file_name="chainsense_template_data.csv",
                    mime="text/csv",
                    help="Use this template to format your data"
                )
        except FileNotFoundError:
            st.warning("Template file not found.")
            
        st.markdown("---")
        
        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload Data (CSV)",
            type=["csv"],
            help="Upload your own supply chain dataset"
        )
        
        if uploaded_file:
            st.success("Custom data loaded!")
        else:
            st.info("Using default dataset")

    # Load Data (Dynamic)
    df = load_data(uploaded_file)
    
    if df is None:
        return 
        
    if df.empty:
        st.warning("Data is empty. Please check your file.")
        return
        
    # Validate Data
    is_valid, validation_msg = validate_data(df)
    
    if not is_valid:
        st.warning(f"⚠️ Column mismatch detected. {validation_msg}. Attempting to fix...")
        
        # Trigger Backend AI Mapping
        with st.spinner("🤖 Attempting Gemma Cloud Semantic Mapping..."):
            mapping = smart_map_columns(df)
        
        if mapping:
            try:
                # Validate the mapping from AI
                validated_mapping = {}
                all_valid = True
                for user_col, target_col in mapping.items():
                    if target_col in TARGET_SCHEMA:
                        validated_mapping[user_col] = target_col
                    else:
                        st.warning(f"AI suggested an invalid target column '{target_col}' for your column '{user_col}'. Ignoring this pair.")
                        all_valid = False
                
                mapping = validated_mapping

                st.success("✨ AI Mapping Generated!")
                with st.expander("View Column Mapping"):
                    st.json(mapping)
                
                df = df.rename(columns=mapping)
                st.info("✅ Columns renamed successfully.")
            except Exception as e:
                st.error(f"Error applying mapping: {e}")
        else:
            st.error("❌ AI could not determine a valid mapping. Proceeding with polyfill.")

        # POLYFILL: Attempt to fill missing critical columns
        df, modified, mods = ensure_critical_columns(df)
        if modified:
            st.warning("⚠️ Some columns were missing. Synthetic values generated:")
            for m in mods:
                st.write(f"- {m}")

    # Final validation check
    is_valid, validation_msg = validate_data(df)
    if not is_valid:
        st.error(f"❌ Critical data still missing after all attempts: {validation_msg}")
        st.info("Please upload a file that contains the required columns or columns that can be easily mapped by the AI.")
        return

    # Title Section
    st.title("🌍 ChainSense: Global Resilience Engine")
    st.markdown("Supply Chain Risk Monitoring powered by Gemma 4")
    st.markdown("---")

    # ---------------------------------------------------------
    # LAYER 1: KPI CARDS
    # ---------------------------------------------------------
    total_orders, on_time_pct, critical_vendors, avg_cost = calculate_kpis(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{total_orders:,}", delta="Orders processed")
    with col2:
        st.metric("On-Time Performance", f"{on_time_pct:.1f}%", delta=f"{100-on_time_pct:.1f}% Late/Issues", delta_color="inverse")
    with col3:
        st.metric("Critical Vendors", f"{critical_vendors}", delta="Risk Score > 75", delta_color="inverse")
    with col4:
        st.metric("Avg Shipping Cost", f"Rp {avg_cost:,.0f}", delta="Per Order")

    st.markdown("###")

    # ---------------------------------------------------------
    # LAYER 2: SPATIAL & TEMPORAL INTELLIGENCE
    # ---------------------------------------------------------
    col_map, col_trend = st.columns([1, 1])
    
    with col_map:
        with st.container(border=True):
            st.subheader("🗺️ Geographic Risk Heatmap")
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Create a simple risk-based map
                map_df = df.groupby(['Customer_Location', 'Latitude', 'Longitude'])['Risk_Score'].mean().reset_index()
                # Rename for st.map compatibility
                map_df = map_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
                
                # Plot map
                st.map(map_df, color='#ff4d4d' if map_df['Risk_Score'].mean() > 50 else '#4da6ff', size=(map_df['Risk_Score'] * 10).tolist())
            else:
                st.info("Coordinates not found. Use Pro-Realism 2.0 dataset to view the map.")

    with col_trend:
        with st.container(border=True):
            st.subheader("📈 Temporal Risk Trend")
            if 'Order_Date' in df.columns:
                trend_df = df.sort_values('Order_Date').groupby('Order_Date')['Risk_Score'].mean().reset_index()
                st.line_chart(trend_df.set_index('Order_Date'), color="#4da6ff")
            else:
                st.info("Temporal data not available.")

    st.markdown("###")

    # ---------------------------------------------------------
    # LAYER 3: CLOUD INTELLIGENCE (EMERGENCY BRIEFING)
    # ---------------------------------------------------------
    with st.container(border=True):
        st.subheader("🚨 Gemma 4: Emergency Strategic Briefing")
        
        col_ai_btn, col_ai_content = st.columns([1, 4])
        analysis_result = st.empty()
        
        with col_ai_btn:
            st.write("Analyze worst-case scenarios and systemic impact.")
            if st.button("Generate Emergency Briefing", type="primary"):
                with col_ai_content:
                    with st.spinner("Gemma is simulating scenarios via Kaggle..."):
                        # Prepare a more dramatic summary
                        avg_risk = df['Risk_Score'].mean()
                        high_risk_vendors = df[df['Risk_Score']>75]['Vendor_Name'].unique().tolist()
                        
                        summary = f"""
                        CURRENT NETWORK SITUATION:
                        - Average Network Risk: {avg_risk:.1f}/100
                        - Critical Vendors (Red Zone): {len(high_risk_vendors)} ({', '.join(high_risk_vendors[:3])})
                        - On-Time Performance: {on_time_pct:.1f}%
                        - Isolated Locations: {', '.join(df[df['Customer_Location']=='Morotai']['Customer_Location'].unique())}
                        
                        INSTRUCTIONS:
                        Provide a short 'Emergency Briefing'. Narrate the worst-case scenario if one of the critical vendors completely fails today. 
                        Mention the estimated impact in percentage (%) and drastic actions that must be taken.
                        """
                        insight = get_gemma_analysis(summary)
                        analysis_result.markdown(insight)
        
        with col_ai_content:
            if not analysis_result.text:
                st.info("Click the button to listen to 'Disruption Storytelling' from Gemma 4.")

    st.markdown("###")

    # ---------------------------------------------------------
    # LAYER 3: DETAIL TABS
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # LAYER 4: DETAIL TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Network Graph", "📋 Scorecard", "⚡ Stress Test", "⚖️ Comparison Tool"])

    # === TAB 1: GRAPH VISUALIZATION ===
    with tab1:
        st.subheader("Supply Chain Network Visualization")
        
        # Filter
        selected_vendors = st.multiselect(
            "Filter Vendor (Node)", 
            options=df['Vendor_Name'].unique(),
            default=None,
            placeholder="Show all vendors..."
        )
        
        # Prepare Graph Data
        graph_df = df.copy()
        if selected_vendors:
            graph_df = graph_df[graph_df['Vendor_Name'].isin(selected_vendors)]
            
        # Create NetworkX Graph
        G = nx.from_pandas_edgelist(graph_df, source='Vendor_Name', target='Customer_Location', edge_attr='Quantity')
        
        # Calculate Centrality Metrics (translated to business terms)
        degree_centrality = nx.degree_centrality(G) # Warehouse Activity
        betweenness_centrality = nx.betweenness_centrality(G) # Bottleneck Point
        
        # Pyvis Network - Dark Mode
        net = Network(height="600px", width="100%", bgcolor="#262730", font_color="white")
        
        # Add nodes with custom attributes
        for node in G.nodes():
            # Determine if node is Vendor or Customer
            is_vendor = node in df['Vendor_Name'].unique()
            
            # Risk coloring for Vendors
            color = "#97c2fc" # Default Blue
            title = f"{node}"
            
            if is_vendor:
                # Get max risk score for this vendor
                vendor_risk = df[df['Vendor_Name'] == node]['Risk_Score'].max()
                
                if vendor_risk > 75:
                    color = "#ff4d4d" # Red - Critical
                elif vendor_risk >= 40:
                    color = "#ffca28" # Yellow - Warning
                else:
                    color = "#00cc66" # Green - Safe
                
                # Business Metrics Tooltip
                activity = degree_centrality.get(node, 0)
                bottleneck = betweenness_centrality.get(node, 0)
                title += f"\nRisk: {vendor_risk:.1f}\nWarehouse Activity: {activity:.2f}\nBottleneck: {bottleneck:.2f}"
                
            net.add_node(node, label=node, title=title, color=color, size=20 if is_vendor else 10)

        # Add edges
        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, value=data['Quantity'])

        # Physics options
        net.barnes_hut()
        
        # Save and display
        try:
            path = ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                path = tmp_file.name
            
            # Save network to the temporary file
            net.save_graph(path)
            
            # Read back the HTML
            with open(path, 'r', encoding='utf-8') as f:
                html_string = f.read()
            
            # Display
            components.html(html_string, height=620)
            
        finally:
            # Clean up
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
            
        st.caption("🔴 Red: High Risk (>75) | 🟡 Yellow: Medium Risk (40-75) | 🟢 Green: Safe (<40)")

    # === TAB 2: VENDOR SCORECARD ===
    with tab2:
        st.subheader("Vendor Performance Scorecard")
        
        # Group by Vendor
        scorecard = df.groupby('Vendor_Name').agg({
            'Order_ID': 'count',
            'Risk_Score': 'mean',
            'Actual_Shipping_Days': 'mean',
            'Delivery_Status': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Unknown'
        }).reset_index()
        
        scorecard.columns = ['Vendor Name', 'Total Orders', 'Avg Risk Score', 'Avg Ship Time (Days)', 'Most Freq Status']
        
        st.dataframe(
            scorecard,
            column_config={
                "Avg Risk Score": st.column_config.ProgressColumn(
                    "Risk Level",
                    help="Average Risk Score (0-100)",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "Most Freq Status": st.column_config.TextColumn(
                    "Delivery Status",
                    help="Most frequent delivery status",
                    validate="^(On Time|Late|Damaged/Returned)$"
                )
            },
            hide_index=True,
            width='stretch'
        )

    # === TAB 3: SIMULATION ===
    with tab3:
        col_sim_control, col_sim_result = st.columns([1, 2])
        
        with col_sim_control:
            st.subheader("⚙️ Stress Test Config")
            st.info("Simulate the impact of external disruptions on vendor risk profiles.")
            
            demand_surge = st.slider("Demand Surge (%)", 0, 100, 0)
            weather_disruption = st.checkbox("Weather Disruption (Flood)", help="Adds estimated delay of 5 days")
            
            st.markdown("---")
            run_gnn = st.button("🧠 Run GNN Prediction (AI)", type="primary", width='stretch')
            
        with col_sim_result:
            st.subheader("📊 Simulation Results")
            
            # Apply logic
            sim_df = df.copy()
            
            # Base risk logic
            sim_df['New_Risk_Score'] = sim_df['Risk_Score'] + (demand_surge * 0.2)
            
            if weather_disruption:
                sim_df['Actual_Shipping_Days'] += 5
                sim_df['New_Risk_Score'] += 20 # Major penalty
            
            sim_df['New_Risk_Score'] = sim_df['New_Risk_Score'].clip(upper=100)
            
            if not run_gnn:
                # Show standard heuristic diff
                sim_df['Risk Increase'] = sim_df['New_Risk_Score'] - sim_df['Risk_Score']
                top_affected = sim_df.groupby('Vendor_Name')[['Risk_Score', 'New_Risk_Score']].mean().reset_index()
                top_affected['Diff'] = top_affected['New_Risk_Score'] - top_affected['Risk_Score']
                top_affected = top_affected.sort_values('Diff', ascending=False).head(5)
                
                st.write("Top 5 Affected Vendors (Heuristic):")
                st.bar_chart(
                    top_affected.set_index('Vendor_Name')[['Risk_Score', 'New_Risk_Score']],
                    color=["#bdc3c7", "#e74c3c"]
                )
                
                avg_risk_before = df['Risk_Score'].mean()
                avg_risk_after = sim_df['New_Risk_Score'].mean()
                st.metric("Proj. Average Network Risk", f"{avg_risk_after:.1f}", f"{avg_risk_after - avg_risk_before:+.1f} points", delta_color="inverse")
                st.info("💡 Click 'Run GNN Prediction' to see analysis from Graph Neural Network & Gemma XAI.")
            else:
                with st.spinner("Building graph & Calling GNN + XAI Gemma from Backend..."):
                    result = get_gnn_prediction(sim_df)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        gnn_data = result.get("gnn_result", {})
                        xai_text = result.get("xai_narrative", "No narrative provided.")
                        high_risk_nodes = result.get("high_risk_nodes", [])
                        
                        st.success(f"✅ GNN Prediction Complete (Model: {gnn_data.get('model_type', 'Unknown')})")
                        
                        st.markdown("### 🤖 XAI: AI Explanation (Gemma 4)")
                        st.info(xai_text)
                        
                        st.markdown("### 🔴 Critical Nodes (Disruption Prob > 60%)")
                        if high_risk_nodes:
                            for node in high_risk_nodes:
                                st.markdown(f"- **{node}**")
                        else:
                            st.success("No critical nodes detected in this scenario.")
                        
                        # Compare original vs GNN risk scores in a dataframe
                        scores = gnn_data.get("risk_scores", [])
                        nodes = gnn_data.get("node_names", list(sim_df['Vendor_Name'].unique())) # fallback if not returned
                        
                        # Note: the API response might not return node_names back directly, but it matches the order we sent
                        # We sent node_names = list(G.nodes()), we can get it back from G.
                        G = nx.from_pandas_edgelist(sim_df, source='Vendor_Name', target='Customer_Location')
                        all_nodes = list(G.nodes())
                        
                        if len(scores) == len(all_nodes):
                            res_df = pd.DataFrame({"Node": all_nodes, "Disruption Probability": scores})
                            # Filter only vendors
                            vendors = sim_df['Vendor_Name'].unique()
                            res_df = res_df[res_df['Node'].isin(vendors)]
                            res_df = res_df.sort_values(by="Disruption Probability", ascending=False).head(10)
                            
                            st.dataframe(
                                res_df, 
                                column_config={
                                    "Disruption Probability": st.column_config.ProgressColumn(
                                        "Probability", format="%.2f", min_value=0, max_value=1
                                    )
                                },
                                hide_index=True, width='stretch'
                            )

    # === TAB 4: COMPARISON TOOL ===
    with tab4:
        st.subheader("⚖️ Side-by-Side Vendor Resilience Comparison")
        st.write("Compare the risk profile and connectivity of two vendors side-by-side.")
        
        col_comp1, col_comp2 = st.columns(2)
        
        all_vendors = sorted(df['Vendor_Name'].unique())
        
        with col_comp1:
            v1 = st.selectbox("Select Vendor A", all_vendors, index=0)
            v1_data = df[df['Vendor_Name'] == v1]
            
            st.metric("Risk Score", f"{v1_data['Risk_Score'].mean():.1f}")
            st.metric("On-Time Rate", f"{(v1_data['Delivery_Status'] == 'On Time').mean()*100:.1f}%")
            st.metric("Avg Lead Time", f"{v1_data['Actual_Shipping_Days'].mean():.1f} days")
            
        with col_comp2:
            v2 = st.selectbox("Select Vendor B", all_vendors, index=1 if len(all_vendors) > 1 else 0)
            v2_data = df[df['Vendor_Name'] == v2]
            
            st.metric("Risk Score", f"{v2_data['Risk_Score'].mean():.1f}", 
                      delta=f"{v2_data['Risk_Score'].mean() - v1_data['Risk_Score'].mean():+.1f}",
                      delta_color="inverse")
            st.metric("On-Time Rate", f"{(v2_data['Delivery_Status'] == 'On Time').mean()*100:.1f}%",
                      delta=f"{(v2_data['Delivery_Status'] == 'On Time').mean()*100 - (v1_data['Delivery_Status'] == 'On Time').mean()*100:+.1f}%")
            st.metric("Avg Lead Time", f"{v2_data['Actual_Shipping_Days'].mean():.1f} days",
                      delta=f"{v2_data['Actual_Shipping_Days'].mean() - v1_data['Actual_Shipping_Days'].mean():+.1f} days",
                      delta_color="inverse")

if __name__ == "__main__":
    main()