# 🔗 ChainSense v2.5 Executive Dashboard

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Gemini AI](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-purple.svg)](https://deepmind.google/technologies/gemini/)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)](#)

**Next-Gen Supply Chain Intelligence Platform**
*Powered by Graph Theory & Generative AI*

</div>

---

## 🌟 What's New in v2.5

ChainSense has been completely re-architected into a professional **Executive Dashboard**. It transforms raw supply chain data into a 3-layer intelligence system.

### 1. 📂 **Universal Data Loader (AI Powered)**
-   **Zero-Config Upload**: Upload ANY csv file.
-   **🤖 AI Semantic Mapper**: If your column names don't match (e.g., "Supplier" instead of "Vendor_Name"), our AI agent (Gemini 1.5) automatically detects and renames them for you.
-   **🛡️ Data Polyfill**: Missing critical data? The system automatically synthesizes missing values (like `Actual_Shipping_Days` or `Risk_Score`) so the dashboard never crashes.

### 2. 📊 **3-Layer Dashboard Architecture**
-   **Layer 1 (KPI Cards)**: Instant view of Total Orders, On-Time Performance, Critical Vendors, and Costs.
-   **Layer 2 (AI Briefing)**: A dedicated Generative AI Consultant that analyzes your unique data and gives 3 "Tactical Recommendations".
-   **Layer 3 (Deep Dive)**: 
    -   **Interactive Risk Map**: Physics-based graph visualization.
    -   **Vendor Scorecard**: Progress bars and status pills.
    -   **Crisis Simulator**: Stress-test your network against Demand Surges and Weather Events.

---

## 🚀 Quick Start

### Prerequisites
-   Python 3.10+
-   A Google Gemini API Key (Get it [here](https://aistudio.google.com/app/apikey))

### Installation

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/yourusername/ChainSense.git
    cd ChainSense
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Secrets**
    Create a file at `.streamlit/secrets.toml`:
    ```toml
    GEMINI_API_KEY = "your-api-key-here"
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🛠️ Features Deep Dive

### 🤖 AI Semantic Mapper
Got a messy dataset? No problem.
-   **Problem**: You have a Pharma dataset with columns like `Distributor_Name` and `City`.
-   **Solution**: ChainSense detects the mismatch, sends the schema to Gemini, and auto-maps it to `Vendor_Name` and `Customer_Location`.

### ⚡ Crisis Simulation
What if demand spikes by 50%? What if a flood hits Jakarta?
-   Use the **Simulation Tab** to adjust sliders.
-   Watch the "Projected Risk" metric update in real-time.
-   See which vendors will collapse under pressure.

### 🗺️ Network Graph
-   **Nodes**: Vendors (colored by Risk) and Customers.
-   **Edges**: Shipment volume.
-   **Physics**: Drag and drop nodes to disentangle complex networks.

---

## 📂 Project Structure

```
ChainSense/
├── app.py                      # Main Application Logic (Monolith)
├── chainsense_synthetic_data.csv # Default Dataset
├── requirements.txt            # Dependencies
├── .streamlit/
│   └── secrets.toml            # API Keys (GitIgnored)
└── README.md                   # Documentation
```

## 📄 License
MIT License. Free to use for educational and enterprise prototypes.

---
<div align="center">
    <i>Built by the ChainSense Team</i>
</div>