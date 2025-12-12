import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai
import tempfile
import os
import json
import random
import numpy as np
import streamlit.components.v1 as components

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="ChainSense | Pro Supply Chain Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

def get_gemini_analysis(summary_text):
    """Call Gemini API for intelligence briefing"""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ API Key not found. Please set GEMINI_API_KEY in secrets.toml."
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Kamu adalah Konsultan Logistik Senior. Berdasarkan ringkasan data berikut, berikan 3 rekomendasi taktis singkat dan actionable untuk manajer gudang.
        Fokus pada mitigasi risiko dan efisiensi biaya.
        
        Data Summary:
        {summary_text}
        
        Output format:
        1. **[Judul Rekomendasi 1]**: [Penjelasan singkat]
        2. **[Judul Rekomendasi 2]**: [Penjelasan singkat]
        3. **[Judul Rekomendasi 3]**: [Penjelasan singkat]
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Analysis Failed: {str(e)}"

def smart_map_columns(df, api_key):
    """
    Use Gemini AI to map user columns to ChainSense standard schema.
    """
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
        'Product_Category': 'Category of product' # Added for better mapping if available
    }
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        user_columns = list(df.columns)
        
        prompt = f"""
        You are a Data Engineering Expert. Map the user's column names to the target standard schema.
        
        Target Schema (Standard): {json.dumps(TARGET_SCHEMA)}
        User Columns (Input): {json.dumps(user_columns)}
        
        Instructions:
        1. Analyze the semantic meaning of User Columns.
        2. Map them to the keys in Target Schema.
        3. Only include mappings where you are confident.
        4. Return ONLY a valid JSON object where keys are User Columns and values are Target Schema keys.
        5. DO NOT wrap the output in markdown code blocks (like ```json). Just the raw JSON string.
        """
        
        response = model.generate_content(prompt)
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        
        mapping = json.loads(text_response)
        return mapping
        
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
        st.warning("⚠️ Column mismatch detected. Attempting AI Semantic Mapping...")
        
        # Check for API Key
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ Data Validation Failed & API Key Missing for AI Fix.\n\n" + validation_msg)
            return

        # Trigger AI Mapping
        with st.spinner("🤖 AI is analyzing column semantics..."):
            mapping = smart_map_columns(df, api_key)
        
        if mapping:
            st.success("✨ AI Mapping Generated!")
            with st.expander("View Column Mapping"):
                st.json(mapping)
            
            # Apply mapping (invert mapping because rename takes {old: new})
            # The AI was asked for {old: new} based on prompt "keys are User Columns"
            # Let's verify prompt: "keys are User Columns and values are Target Schema keys" -> {Old: New} -> Correct for rename
            
            try:
                df = df.rename(columns=mapping)
                st.info("✅ Columns renamed successfully.")
                
                # POLYFILL: Attempt to fill missing critical data
                df, modified, mods = ensure_critical_columns(df)
                if modified:
                    st.warning("⚠️ Some columns were still missing. Synthetic values generated:")
                    for m in mods:
                        st.write(f"- {m}")

                # Re-validate
                is_valid_2, validation_msg_2 = validate_data(df)
                if not is_valid_2:
                    st.error(f"❌ Critical data still missing: {validation_msg_2}")
                    return
            except Exception as e:
                st.error(f"Error applying mapping: {e}")
                return
        else:
            st.error("❌ AI could not determine a valid mapping.")
            st.code(validation_msg)
            return

    # Title Section
    st.title("🔗 ChainSense Executive Dashboard")
    st.markdown("Supply Chain Risk Monitoring & Intelligence System")
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
    # LAYER 2: AI INTELLIGENCE
    # ---------------------------------------------------------
    with st.container(border=True):
        st.subheader("🤖 ChainSense Intelligence Briefing")
        
        col_ai_btn, col_ai_content = st.columns([1, 4])
        
        analysis_result = st.empty()
        
        with col_ai_btn:
            st.write("Generate AI-powered insights based on current risk metrics.")
            if st.button("Generate AI Analysis", type="primary", use_container_width=True):
                with col_ai_content:
                    with st.spinner("Consulting AI Expert..."):
                        # Prepare summary for AI
                        summary = f"""
                        Total Orders: {total_orders}
                        On-Time Rate: {on_time_pct:.1f}%
                        Critical Vendors Count: {critical_vendors}
                        Average Shipping Cost: {avg_cost}
                        Top High Risk Vendors: {', '.join(df[df['Risk_Score']>75]['Vendor_Name'].unique().tolist()[:5])}
                        Average Risk Score: {df['Risk_Score'].mean():.1f}
                        """
                        insight = get_gemini_analysis(summary)
                        analysis_result.markdown(insight)
        
        with col_ai_content:
            if not analysis_result.text:
                st.info("Click the button to generate a strategic briefing.")

    st.markdown("###")

    # ---------------------------------------------------------
    # LAYER 3: DETAIL TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🗺️ Peta Risiko (Graph)", "📋 Vendor Scorecard", "⚡ Simulasi Krisis"])

    # === TAB 1: GRAPH VISUALIZATION ===
    with tab1:
        st.subheader("Supply Chain Network Visualization")
        
        # Filter
        selected_vendors = st.multiselect(
            "Filter Vendor (Node)", 
            options=df['Vendor_Name'].unique(),
            default=None,
            placeholder="Tampilkan semua vendor..."
        )
        
        # Prepare Graph Data
        graph_df = df.copy()
        if selected_vendors:
            graph_df = graph_df[graph_df['Vendor_Name'].isin(selected_vendors)]
            
        # Create NetworkX Graph
        G = nx.from_pandas_edgelist(graph_df, source='Vendor_Name', target='Customer_Location', edge_attr='Quantity')
        
        # Calculate Centrality Metrics (translated to business terms)
        degree_centrality = nx.degree_centrality(G) # Aktivitas Gudang
        betweenness_centrality = nx.betweenness_centrality(G) # Titik Rawan Macet
        
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
                title += f"\nRisiko: {vendor_risk:.1f}\nAktivitas Gudang: {activity:.2f}\nTitik Rawan Macet: {bottleneck:.2f}"
                
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
            
        st.caption("🔴 Merah: Risiko Tinggi (>75) | 🟡 Kuning: Risiko Sedang (40-75) | 🟢 Hijau: Aman (<40)")

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
            use_container_width=True
        )

    # === TAB 3: SIMULATION ===
    with tab3:
        col_sim_control, col_sim_result = st.columns([1, 2])
        
        with col_sim_control:
            st.subheader("⚙️ Stress Test Config")
            st.info("Simulasi dampak gangguan eksternal terhadap profil risiko vendor.")
            
            demand_surge = st.slider("Kenaikan Permintaan (%)", 0, 100, 0)
            weather_disruption = st.checkbox("Gangguan Cuaca (Banjir)", help="Menambah estimasi delay 5 hari")
            
        with col_sim_result:
            st.subheader("📊 Hasil Simulasi")
            
            # Apply logic
            sim_df = df.copy()
            
            # Base risk logic from original (simplified for simulation demo)
            # If demand surges, risk increases slightly (0.1 point per % for demo)
            sim_df['New_Risk_Score'] = sim_df['Risk_Score'] + (demand_surge * 0.2)
            
            # If weather disruption, add 5 days to metric and spike risk
            if weather_disruption:
                sim_df['Actual_Shipping_Days'] += 5
                sim_df['New_Risk_Score'] += 20 # Major penalty
            
            # Cap at 100
            sim_df['New_Risk_Score'] = sim_df['New_Risk_Score'].clip(upper=100)
            
            # Show Before/After Comparison for top 5 affected
            sim_df['Risk Increase'] = sim_df['New_Risk_Score'] - sim_df['Risk_Score']
            top_affected = sim_df.groupby('Vendor_Name')[['Risk_Score', 'New_Risk_Score']].mean().reset_index()
            top_affected['Diff'] = top_affected['New_Risk_Score'] - top_affected['Risk_Score']
            top_affected = top_affected.sort_values('Diff', ascending=False).head(5)
            
            st.write("Top 5 Vendor Terdampak:")
            
            # Chart
            st.bar_chart(
                top_affected.set_index('Vendor_Name')[['Risk_Score', 'New_Risk_Score']],
                color=["#bdc3c7", "#e74c3c"]
            )
            
            avg_risk_before = df['Risk_Score'].mean()
            avg_risk_after = sim_df['New_Risk_Score'].mean()
            
            st.metric("Proj. Average Network Risk", f"{avg_risk_after:.1f}", f"{avg_risk_after - avg_risk_before:+.1f} points", delta_color="inverse")

if __name__ == "__main__":
    main()