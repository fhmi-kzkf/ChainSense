# ============================================================================
# 🌍 ChainSense Edge — Kaggle Backend API
# ============================================================================
# Run this ENTIRE script as a single cell in a Kaggle Notebook
# with GPU T4 accelerator enabled.
#
# Architecture:
#   [Streamlit Frontend] --(Ngrok Tunnel)--> [FastAPI Backend on Kaggle]
#       ├── /semantic-map   → Gemma 4 column mapping
#       ├── /analyze-risk   → Gemma 4 tactical briefing
#       └── /analyze        → Unified endpoint (mapping + analysis + GNN)
#
# Models:
#   - LLM:  unsloth/gemma-4-e2b-it-bnb-4bit (Quantized for T4 GPU)
#   - GNN:  Custom 2-layer GCN (PyTorch Geometric)
# ============================================================================

# ============================================================================
# CELL 1: INSTALLATION (Jalankan cell ini pertama kali di Colab / Kaggle)
# ============================================================================
# Copy baris di bawah ini ke Cell pertama di Colab Anda:

# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
# !pip install fastapi uvicorn pyngrok nest_asyncio
# !pip install torch-geometric

# ============================================================================
# CELL 2: BACKEND CODE (Jalankan cell ini setelah instalasi selesai)
# ============================================================================
# ⚠️ PENTING: Unsloth harus di-import SEBELUM torch dan library lainnya
try:
    import unsloth
except ImportError:
    pass

import os
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import nest_asyncio
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Apply nest_asyncio so uvicorn can run inside Jupyter
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chainsense-backend")

# ============================================================================
# SECTION 1: GNN ENGINE (PyTorch Geometric)
# ============================================================================
# A lightweight 2-layer Graph Convolutional Network for node-level
# disruption risk prediction. It takes adjacency + node features from
# the frontend's GraphAnalyzer and outputs a risk probability per node.
# ============================================================================

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    PYGEOMETRIC_AVAILABLE = True
    logger.info("✅ PyTorch Geometric loaded.")
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    logger.warning("⚠️ PyTorch Geometric not available. GNN features disabled.")


class GCNRiskPredictor(torch.nn.Module):
    """
    2-Layer Graph Convolutional Network for supply chain risk prediction.
    
    Input Features (per node):
        - degree_centrality      (float)
        - betweenness_centrality (float)
        - risk_score_normalized  (float, 0-1)
        - avg_shipping_days_norm (float, 0-1)
        - on_time_ratio          (float, 0-1)
    
    Output:
        - disruption_probability (float, 0-1) per node
    """
    
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # Binary output
        self.dropout = torch.nn.Dropout(p=0.3)
    
    def forward(self, x, edge_index):
        # Layer 1: Conv + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2: Conv + Sigmoid (probability)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        
        return x.squeeze(-1)  # (num_nodes,)


class GNNPredictor:
    """
    Wrapper class that handles building PyG Data objects from raw
    adjacency/feature dictionaries sent by the Streamlit frontend,
    running inference, and returning risk scores.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if PYGEOMETRIC_AVAILABLE:
            self.model = GCNRiskPredictor(in_channels=5, hidden_channels=16).to(self.device)
            self.model.eval()
            logger.info(f"🧠 GNN initialized on {self.device}")
        else:
            self.model = None
            logger.warning("GNN model not initialized (PyG missing).")
    
    def predict(self, node_features: List[List[float]], edge_list: List[List[int]]) -> Dict[str, Any]:
        """
        Run GNN inference.

        Args:
            node_features: List of [degree_c, betweenness_c, risk_norm, ship_norm, ontime_ratio]
            edge_list:     List of [source_idx, target_idx] pairs

        Returns:
            dict with 'risk_scores' (list of floats) and 'high_risk_indices'
        """
        if not PYGEOMETRIC_AVAILABLE or self.model is None:
            # Fallback: heuristic-based risk estimation
            return self._heuristic_fallback(node_features)
        
        try:
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            
            if not edge_list:
                # No edges → isolated nodes → return feature-based risk
                return self._heuristic_fallback(node_features)
            
            # edge_index shape: [2, num_edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
            
            data = Data(x=x, edge_index=edge_index)
            
            with torch.no_grad():
                risk_probs = self.model(data.x, data.edge_index)
            
            scores = risk_probs.cpu().numpy().tolist()
            high_risk = [i for i, s in enumerate(scores) if s > 0.6]
            
            return {
                "risk_scores": [round(s, 4) for s in scores],
                "high_risk_indices": high_risk,
                "model_type": "GCN",
                "device": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"GNN inference failed: {e}")
            return self._heuristic_fallback(node_features)
    
    def _heuristic_fallback(self, node_features: List[List[float]]) -> Dict[str, Any]:
        """
        Fallback risk scoring when GNN is unavailable.
        Uses a weighted combination of input features.
        """
        scores = []
        for feats in node_features:
            if len(feats) >= 5:
                # Weighted: risk_score(0.4) + betweenness(0.25) + shipping(0.2) + (1-ontime)(0.15)
                score = (feats[2] * 0.4) + (feats[1] * 0.25) + (feats[3] * 0.2) + ((1 - feats[4]) * 0.15)
            elif len(feats) >= 3:
                score = feats[2] * 0.6 + feats[1] * 0.4
            else:
                score = 0.5
            scores.append(round(min(max(score, 0.0), 1.0), 4))
        
        high_risk = [i for i, s in enumerate(scores) if s > 0.6]
        
        return {
            "risk_scores": scores,
            "high_risk_indices": high_risk,
            "model_type": "heuristic_fallback",
            "device": "cpu"
        }


# ============================================================================
# SECTION 2: LLM ENGINE (Gemma 4 via Unsloth)
# ============================================================================

print("=" * 60)
print("🚀 Loading Gemma 4 E2B model via Unsloth...")
print("   This may take 2-3 minutes on first run.")
print("=" * 60)

try:
    from unsloth import FastLanguageModel

    MODEL_NAME = "unsloth/gemma-4-e2b-it-bnb-4bit"
    MAX_SEQ_LENGTH = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    LLM_READY = True
    print("✅ Gemma 4 E2B loaded successfully!")
    
except Exception as e:
    LLM_READY = False
    model = None
    tokenizer = None
    print(f"❌ Failed to load Gemma 4: {e}")
    print("   LLM features will return placeholder responses.")


def generate_response(prompt: str, max_tokens: int = 1536) -> str:
    """
    Generate text from Gemma 4 using the Unsloth-optimized pipeline.
    """
    if not LLM_READY:
        return "⚠️ LLM not loaded. Please check Kaggle GPU and Unsloth installation."
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode only the generated portion
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response.strip()
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return f"❌ Generation error: {str(e)}"


# ============================================================================
# SECTION 3: FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="ChainSense Edge Backend",
    description="Gemma 4 + GNN powered supply chain risk API",
    version="1.0.0"
)

# Allow all CORS for Streamlit/Ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GNN
gnn_predictor = GNNPredictor()


# --- Pydantic Models ---

class SemanticMapRequest(BaseModel):
    task: str = "mapping"
    target_schema: Dict[str, str]
    user_columns: List[str]

class AnalyzeRiskRequest(BaseModel):
    task: str = "analysis"
    summary: str

class UnifiedRequest(BaseModel):
    """Unified request model for the /analyze endpoint."""
    task: str  # "mapping" | "analysis" | "gnn_predict"
    # For mapping
    target_schema: Optional[Dict[str, str]] = None
    user_columns: Optional[List[str]] = None
    # For analysis
    summary: Optional[str] = None
    # For GNN
    node_features: Optional[List[List[float]]] = None
    edge_list: Optional[List[List[int]]] = None
    node_names: Optional[List[str]] = None


# --- Endpoint: /semantic-map ---

@app.post("/semantic-map")
async def semantic_map(req: SemanticMapRequest):
    """Map user CSV columns to ChainSense standard schema using Gemma 4."""
    
    prompt = f"""You are a Data Engineering Expert for supply chain systems.
Map the user's CSV column names to the target standard schema.

Target Schema: {json.dumps(req.target_schema)}
User Columns: {json.dumps(req.user_columns)}

Instructions:
1. Analyze the semantic meaning of each user column.
2. Map each to the most appropriate key in the Target Schema.
3. Only include confident mappings.
4. Return ONLY a valid JSON object where keys are User Columns and values are Target Schema keys.
5. Do NOT include any explanation, markdown, or code blocks. Just raw JSON."""

    raw = generate_response(prompt, max_tokens=300)
    
    # Parse JSON from response
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        mapping = json.loads(clean)
    except json.JSONDecodeError:
        mapping = {}
        logger.warning(f"Failed to parse mapping JSON: {raw[:200]}")
    
    return {"mapping": mapping, "raw_response": raw}


# --- Endpoint: /analyze-risk ---

@app.post("/analyze-risk")
async def analyze_risk(req: AnalyzeRiskRequest):
    """Generate tactical resilience briefing from supply chain metrics."""
    
    prompt = f"""You are a Global Supply Chain Resilience Expert.
Provide an IN-DEPTH, DETAILED, and STRUCTURED "Emergency Strategic Briefing". Ignore any previous instructions to keep it short.
CRITICAL INSTRUCTION: You MUST write your entire response in ENGLISH ONLY. Do NOT use Indonesian.

Data Summary:
{req.summary}

Mandatory Report Format:
## 🔍 Critical Situation Analysis
[In-depth analysis of current logistics system vulnerabilities]

## 🚨 Worst-Case Scenario & Impact Estimation
[Specific scenario of cascading failure and percentage impact on operations/financials]

## ⚡ Drastic Mitigation Steps
1. **[Action 1]**: [Technical execution details]
2. **[Action 2]**: [Technical execution details]
3. **[Action 3]**: [Technical execution details]

## 🎯 Priority Execution Command
[One most urgent actionable sentence]"""

    response = generate_response(prompt, max_tokens=1536)
    return {"response": response}


# --- Endpoint: /analyze (Unified — used by Streamlit app.py) ---

@app.post("/analyze")
async def unified_analyze(req: UnifiedRequest):
    """
    Unified endpoint that handles all tasks:
      - "mapping"      → Column semantic mapping
      - "analysis"     → Risk briefing
      - "gnn_predict"  → GNN disruption prediction + XAI narrative
    """
    
    # ── TASK: COLUMN MAPPING ──
    if req.task == "mapping":
        if not req.target_schema or not req.user_columns:
            raise HTTPException(status_code=400, detail="Missing target_schema or user_columns")
        
        prompt = f"""You are a Data Engineering Expert for supply chain systems.
Your task is to map the user's column names to a standard schema.

**Target Schema:**
```json
{json.dumps(req.target_schema, indent=2)}
```

**User's Columns:**
{json.dumps(req.user_columns)}

Instructions:
1. Analyze the semantic meaning of each user column.
2. Map each user column to the most appropriate key from the **Target Schema**.
3. The JSON output's **keys** must be the **User's Columns**.
4. The JSON output's **values** must be the corresponding keys from the **Target Schema**.
5. Only include mappings you are confident about. If a user column does not match any schema key, omit it.
6. Return ONLY a valid JSON object. Do NOT include any explanation, markdown, or code blocks.

**Example:**
If the user's columns are `["Distributor", "City", "Date"]`, the correct output is:
```json
{{
  "Distributor": "Vendor_Name",
  "City": "Customer_Location",
  "Date": "Order_Date"
}}
```"""

        raw = generate_response(prompt, max_tokens=300)
        
        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            mapping = json.loads(clean)
        except json.JSONDecodeError:
            mapping = {}
        
        return {"mapping": mapping, "raw_response": raw}
    
    # ── TASK: RISK ANALYSIS BRIEFING ──
    elif req.task == "analysis":
        if not req.summary:
            raise HTTPException(status_code=400, detail="Missing summary text")
        
        prompt = f"""You are a Global Supply Chain Resilience Expert.
Provide an IN-DEPTH, DETAILED, and STRUCTURED "Emergency Strategic Briefing". Ignore any previous instructions to keep it short.
CRITICAL INSTRUCTION: You MUST write your entire response in ENGLISH ONLY. Do NOT use Indonesian.

Data Summary:
{req.summary}

Mandatory Report Format:
## 🔍 Critical Situation Analysis
[In-depth analysis of current logistics system vulnerabilities]

## 🚨 Worst-Case Scenario & Impact Estimation
[Specific scenario of cascading failure and percentage impact on operations/financials]

## ⚡ Drastic Mitigation Steps
1. **[Action 1]**: [Technical execution details]
2. **[Action 2]**: [Technical execution details]
3. **[Action 3]**: [Technical execution details]

## 🎯 Priority Execution Command
[One most urgent actionable sentence]"""
        
        response = generate_response(prompt, max_tokens=1536)
        return {"response": response}
    
    # ── TASK: GNN PREDICTION + XAI ──
    elif req.task == "gnn_predict":
        if not req.node_features:
            raise HTTPException(status_code=400, detail="Missing node_features for GNN")
        
        edge_list = req.edge_list or []
        node_names = req.node_names or [f"Node_{i}" for i in range(len(req.node_features))]
        
        # Step 1: Run GNN prediction
        gnn_result = gnn_predictor.predict(req.node_features, edge_list)
        
        # Step 2: Build context for XAI narrative
        risk_summary_parts = []
        for i, name in enumerate(node_names):
            score = gnn_result["risk_scores"][i] if i < len(gnn_result["risk_scores"]) else 0.5
            level = "🔴 CRITICAL" if score > 0.7 else "🟡 WARNING" if score > 0.4 else "🟢 SAFE"
            risk_summary_parts.append(f"- {name}: {score:.2%} ({level})")
        
        risk_context = "\n".join(risk_summary_parts[:15])  # Limit to top 15 nodes
        
        high_risk_names = [node_names[i] for i in gnn_result["high_risk_indices"] if i < len(node_names)]
        
        # Step 3: Generate XAI narrative with Gemma 4
        xai_prompt = f"""You are an Explainable AI analyst for supply chain networks.
A Graph Neural Network has predicted the following disruption risk scores for each node in the network:

{risk_context}

High-risk nodes: {', '.join(high_risk_names) if high_risk_names else 'None detected'}
Model used: {gnn_result['model_type']}

Based on these predictions, provide:
1. A brief explanation of WHY these nodes are high-risk (based on their network position).
2. The most vulnerable supply chain path.
3. One actionable recommendation to reduce systemic risk.

Keep your response concise (under 150 words)."""
        
        xai_narrative = generate_response(xai_prompt, max_tokens=250)
        
        return {
            "gnn_result": gnn_result,
            "high_risk_nodes": high_risk_names,
            "xai_narrative": xai_narrative
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task: '{req.task}'. Use 'mapping', 'analysis', or 'gnn_predict'.")


# --- Health Check ---

@app.get("/")
async def health():
    return {
        "status": "online",
        "service": "ChainSense Edge Backend",
        "llm_ready": LLM_READY,
        "llm_model": MODEL_NAME if LLM_READY else "not loaded",
        "gnn_ready": PYGEOMETRIC_AVAILABLE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }


# ============================================================================
# SECTION 4: START SERVER WITH NGROK TUNNEL
# ============================================================================

NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ← Paste your Ngrok auth token
PORT = 8000

def start_server():
    """Start Uvicorn in a background thread (Jupyter-compatible)."""
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

if __name__ == "__main__" or True:  # Always run in notebook context
    
    # Start FastAPI in background thread
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Setup Ngrok tunnel
    try:
        from pyngrok import ngrok, conf
        
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        
        # Kill any existing tunnels
        tunnels = ngrok.get_tunnels()
        for t in tunnels:
            ngrok.disconnect(t.public_url)
        
        # Open new tunnel
        public_url = ngrok.connect(PORT, "http").public_url
        
        print("\n" + "=" * 60)
        print("🌍 CHAINSENSE EDGE BACKEND IS LIVE!")
        print("=" * 60)
        print(f"🔗 Public URL  : {public_url}")
        print(f"🏠 Local URL   : http://localhost:{PORT}")
        print(f"📡 Health Check: {public_url}/")
        print(f"📡 API Docs    : {public_url}/docs")
        print("=" * 60)
        print()
        print("📋 COPY THIS URL INTO YOUR app.py:")
        print(f'   BACKEND_URL = "{public_url}"')
        print()
        print("=" * 60)
        print("⏳ Server is running. Keep this notebook open.")
        print("   Press STOP in Kaggle to terminate.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Ngrok tunnel failed: {e}")
        print(f"   Server is still running locally at http://localhost:{PORT}")
        print("   Get a free Ngrok token at https://dashboard.ngrok.com/signup")
