# 🔗 ChainSense — AI-Powered Supply Chain Resilience Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Gemma 4](https://img.shields.io/badge/AI-Gemma%204%20E2B-purple.svg)](https://huggingface.co/google/gemma-4)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Predict vendor failures before they happen.**  
*Powered by Gemma 4 + Graph Neural Networks*

</div>

---

## 🌍 What is ChainSense?

ChainSense is a real-time supply chain intelligence platform that transforms raw logistics data into strategic AI-driven decisions. It combines:

- 🤖 **Gemma 4** (via Unsloth, 4-bit quantized) for natural language reasoning, emergency briefings, and explainable AI
- 🧠 **Graph Convolutional Network (GCN)** for node-level disruption probability prediction
- 📊 **Interactive Streamlit Dashboard** for live risk monitoring and scenario simulation

---

## 🏗️ Architecture

```
[Streamlit Frontend]
       │
       │  HTTP (via Ngrok tunnel)
       ▼
[Kaggle Backend — FastAPI]
       ├── /analyze  (task: "mapping")      → AI Column Semantic Mapper
       ├── /analyze  (task: "analysis")     → Emergency Strategic Briefing
       └── /analyze  (task: "gnn_predict")  → GCN Risk Prediction + XAI Narrative
```

---

## ✨ Features

### 🤖 Universal AI Data Loader
Upload **any CSV file** — ChainSense handles the rest.
- **AI Semantic Mapper**: Gemma 4 auto-detects and renames mismatched columns (e.g., `"Distributor"` → `"Vendor_Name"`)
- **Smart Polyfill**: Synthesizes missing critical columns so the dashboard never crashes

### 📊 KPI Dashboard (Layer 1)
Instant view of your network's vital signs:
- **Total Orders** — Volume processed
- **On-Time Performance (%)** — Delivery reliability
- **Critical Vendors** — Vendors with Risk Score > 75
- **Avg Shipping Cost** — Cost per order

### 🗺️ Spatial & Temporal Intelligence (Layer 2)
- **Geographic Risk Heatmap** — Map-based risk view (requires lat/lon columns)
- **Temporal Risk Trend** — Risk score trend over time

### 🚨 Emergency Strategic Briefing (AI)
One click → Gemma 4 generates a structured crisis report:
- Critical situation analysis
- Worst-case cascading failure scenario + estimated % impact
- 3 drastic mitigation steps
- Priority execution command

### 🕸️ Supply Chain Network Graph
Physics-based interactive graph of your vendor–customer network:
- 🔴 **Red** = Critical Risk (>75) | 🟡 **Yellow** = Warning (40–75) | 🟢 **Green** = Safe (<40)
- Hover tooltips: Risk Score, Warehouse Activity, Bottleneck Score
- Filter by vendor

### 📋 Vendor Scorecard
Per-vendor performance table with progress bar risk level, lead time, and delivery status.

### ⚡ Stress Test / Crisis Simulator
Simulate external disruptions and measure network impact:
- **Demand Surge slider** (0–100%)
- **Weather Disruption toggle** (+5 days delay, +20 risk penalty)
- **Heuristic mode**: Instant top-5 affected vendor chart
- **GNN mode**: Run Graph Convolutional Network → disruption probability per node + Gemma 4 XAI narrative

### ⚖️ Vendor Comparison Tool
Side-by-side comparison of any two vendors — Risk Score, On-Time Rate, Avg Lead Time, with automatic delta.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A running ChainSense Kaggle Backend (see [Backend Setup](#backend-setup))

### 1. Clone & Install

```bash
git clone https://github.com/fhmi-kzkf/ChainSense.git
cd ChainSense
pip install -r requirements.txt
```

### 2. Configure Backend URL

In `app.py`, set your Ngrok URL from the Kaggle backend:

```python
BACKEND_URL = "https://your-ngrok-url.ngrok-free.app"
```

### 3. Run the App

```bash
streamlit run app.py
# or on Windows:
run_app.bat
```

---

## 🖥️ Backend Setup (Kaggle + Ngrok)

The AI features require the Kaggle backend running on a GPU notebook.

### Option A — Gemma 4 + GNN (Recommended)
Use `kaggle_backend.py` — Full featured: Gemma 4 E2B (Unsloth 4-bit) + 2-layer GCN

### Option B — Ollama + Gemma 2
Use `kaggle_backend_ollama.py` — Lighter setup using Ollama inference engine

### Option C — HuggingFace Transformers
Use `kaggle_backend_transformers.py` — Uses standard transformers pipeline, no Unsloth required

**Steps:**
1. Open a Kaggle notebook with **GPU T4** enabled
2. Paste the chosen backend file as a single cell
3. Install dependencies (see comments inside each backend file)
4. Set your `NGROK_AUTH_TOKEN` and run
5. Copy the printed `Public URL` into `app.py`'s `BACKEND_URL`

---

## 📁 Project Structure

```
ChainSense/
├── app.py                          # Main Streamlit application
├── kaggle_backend.py               # Backend: Gemma 4 + GCN (FastAPI)
├── kaggle_backend_ollama.py        # Backend: Ollama + Gemma 2 (FastAPI)
├── kaggle_backend_transformers.py  # Backend: HuggingFace Transformers (FastAPI)
├── generate_chainsense_data.py     # Synthetic supply chain data generator
├── chainsense_synthetic_data.csv   # Default dataset
├── chainsense_pharma_data.csv      # Alternative: Pharma supply chain dataset
├── Chainsense_Routes.csv           # Sample logistics routes data
├── requirements.txt                # Python dependencies
├── config.ini                      # App configuration
├── run_app.bat                     # Windows quick-run script
├── setup.bat                       # Windows setup script
├── runtime.txt                     # Python version spec (for deployment)
└── .streamlit/
    └── secrets.toml                # API keys (gitignored)
```

---

## 🧠 Technical Details

### GNN Architecture
- **Model**: 2-layer Graph Convolutional Network (GCNConv via PyTorch Geometric)
- **Input features per node**: degree centrality, betweenness centrality, normalized risk score, normalized shipping days, on-time ratio
- **Output**: Disruption probability (0–1) per node
- **Fallback**: Weighted heuristic scoring when PyG is unavailable

### LLM (Gemma 4 E2B)
- Loaded via **Unsloth** with 4-bit quantization for GPU efficiency
- Tasks: semantic column mapping, emergency briefings, XAI narrative generation
- All prompts are English-only, structured output format

---

## 📄 License

MIT License — Free to use for educational and enterprise prototypes.

---

<div align="center">
  <b>ChainSense</b> — See the Break Before It Breaks.<br/>
  <i>Built for the Gemma Hackathon Special Track</i>
</div>
