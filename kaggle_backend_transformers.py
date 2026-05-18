# ============================================================================
# 🌍 ChainSense Edge — Colab/Kaggle Backend API (Transformers Version)
# ============================================================================
import os
import json
import asyncio
import logging
import subprocess
import time
from threading import Thread
from typing import Optional, List, Dict, Any

# --- AUTO-INSTALLER SECTION ---
def install_dependencies():
    print("📦 Checking & Installing system dependencies...")
    # 1. Install system dependencies for bitsandbytes
    os.system("apt-get update -y && apt-get install -y nvidia-cuda-toolkit")
    
    # 2. Install Python libraries
    # Using unsloth for faster gemma-2-2b inference if available, otherwise standard transformers
    os.system("pip install -q -U transformers accelerate bitsandbytes fastapi uvicorn pyngrok nest_asyncio torch-geometric")
    print("✅ Installation complete.")

# Jalankan instalasi otomatis jika di environment cloud
if os.path.exists('/content') or os.path.exists('/kaggle/working'):
    install_dependencies()

import torch
import torch.nn.functional as F
import numpy as np
import nest_asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Apply nest_asyncio so uvicorn can run inside Jupyter
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chainsense-transformers-backend")

# ============================================================================
# SECTION 1: GNN ENGINE (PyTorch Geometric)
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
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(p=0.3)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        return x.squeeze(-1)


class GNNPredictor:
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
        if not PYGEOMETRIC_AVAILABLE or self.model is None:
            return self._heuristic_fallback(node_features)
        
        try:
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            if not edge_list: return self._heuristic_fallback(node_features)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
            
            with torch.no_grad():
                risk_probs = self.model(x, edge_index)
            
            scores = risk_probs.cpu().numpy().tolist()
            return {
                "risk_scores": [round(s, 4) for s in scores],
                "high_risk_indices": [i for i, s in enumerate(scores) if s > 0.6],
                "model_type": "GCN",
                "device": str(self.device)
            }
        except Exception as e:
            logger.error(f"GNN inference failed: {e}")
            return self._heuristic_fallback(node_features)
    
    def _heuristic_fallback(self, node_features: List[List[float]]) -> Dict[str, Any]:
        scores = []
        for feats in node_features:
            if len(feats) >= 5:
                score = (feats[2] * 0.4) + (feats[1] * 0.25) + (feats[3] * 0.2) + ((1 - feats[4]) * 0.15)
            else: score = 0.5
            scores.append(round(min(max(score, 0.0), 1.0), 4))
        
        return {
            "risk_scores": scores,
            "high_risk_indices": [i for i, s in enumerate(scores) if s > 0.6],
            "model_type": "heuristic_fallback",
            "device": "cpu"
        }


# ============================================================================
# SECTION 2: LLM ENGINE (Gemma via Transformers)
# ============================================================================
print("=" * 60)
print("🚀 Loading Gemma 4 2B model via Hugging Face Transformers...")
print("=" * 60)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    MODEL_NAME = "google/gemma-4-E2B-it"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    LLM_READY = True
    print("✅ Gemma loaded successfully!")
except Exception as e:
    LLM_READY = False
    model = None
    tokenizer = None
    print(f"❌ Failed to load Gemma: {e}")


def generate_response(prompt: str, max_tokens: int = 1536) -> str:
    if not LLM_READY: return "⚠️ LLM not loaded."
    try:
        chat_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        # Bersihkan token penanda akhir jika bocor dari model
        return response.replace("<end_of_turn>", "").replace("<eos>", "").strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# SECTION 3: FASTAPI APPLICATION
# ============================================================================
app = FastAPI(title="ChainSense Transformers Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
gnn_predictor = GNNPredictor()

class UnifiedRequest(BaseModel):
    task: str
    target_schema: Optional[Dict[str, str]] = None
    user_columns: Optional[List[str]] = None
    summary: Optional[str] = None
    node_features: Optional[List[List[float]]] = None
    edge_list: Optional[List[List[int]]] = None
    node_names: Optional[List[str]] = None

@app.post("/analyze")
async def analyze(req: UnifiedRequest):
    if req.task == "mapping":
        prompt = f"""You are a Data Engineering Expert for supply chain systems.
Your task is to map the user's column names to a standard schema.

**Target Schema:**
```json
{json.dumps(req.target_schema, indent=2)}
```

**User's Columns:**
{json.dumps(req.user_columns)}

**Instructions:**
1.  Analyze the semantic meaning of each user column.
2.  Map each user column to the most appropriate key from the **Target Schema**.
3.  The JSON output's **keys** must be the **User's Columns**.
4.  The JSON output's **values** must be the corresponding keys from the **Target Schema**.
5.  Only include mappings you are confident about. If a user column does not match any schema key, omit it.
6.  Return ONLY a valid JSON object. Do NOT include any explanation, markdown, or code blocks.

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
            import re
            match = re.search(r'(\{.*\})', raw, re.DOTALL)
            return {"mapping": json.loads(match.group(1)) if match else json.loads(raw)}
        except: return {"mapping": {}}
    
    elif req.task == "analysis":
        prompt = f"""You are a Global Supply Chain Resilience Expert.
Provide an IN-DEPTH, DETAILED, and STRUCTURED "Emergency Strategic Briefing". Ignore any previous instructions to keep it short.
CRITICAL INSTRUCTION: You MUST write your entire response in ENGLISH ONLY. Do NOT use Indonesian, even if the data contains Indonesian names.

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
        return {"response": generate_response(prompt, max_tokens=1536)}
    
    elif req.task == "gnn_predict":
        gnn_res = gnn_predictor.predict(req.node_features, req.edge_list)
        node_names = req.node_names or [f"Node_{i}" for i in range(len(req.node_features))]
        xai_prompt = f"Explain the vulnerability risk of this network based on these scores: {gnn_res['risk_scores'][:10]}"
        return {
            "gnn_result": gnn_res,
            "high_risk_nodes": [node_names[i] for i in gnn_res["high_risk_indices"] if i < len(node_names)],
            "xai_narrative": generate_response(xai_prompt, max_tokens=250)
        }
    
    raise HTTPException(status_code=400, detail="Unknown task")

# --- SECURE TOKEN MANAGEMENT ---
NGROK_AUTH_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    NGROK_AUTH_TOKEN = user_secrets.get_secret("NGROK_TOKEN")
    print("🔐 Token Ngrok dimuat dari Secrets.")
except Exception:
    import getpass
    if os.environ.get('NGROK_TOKEN'):
        NGROK_AUTH_TOKEN = os.environ.get('NGROK_TOKEN')
    else:
        NGROK_AUTH_TOKEN = getpass.getpass("Masukkan Ngrok Auth Token Anda: ")

PORT = 8000
if __name__ == "__main__":
    from pyngrok import ngrok, conf
    if not NGROK_AUTH_TOKEN:
        print("❌ Error: Token Ngrok diperlukan.")
    else:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT), daemon=True).start()
        try:
            public_url = ngrok.connect(PORT, "http").public_url
            print(f"\n" + "="*60)
            print(f"🌍 TRANSFORMERS BACKEND LIVE: {public_url}")
            print(f"📋 Salin URL di atas ke BACKEND_URL di app.py Anda")
            print("="*60)
            
            print("⏳ Server is running. Keep this Kaggle cell running.")
            while True:
                time.sleep(3600)
        except Exception as e:
            print(f"❌ Gagal membuka tunnel: {e}")
