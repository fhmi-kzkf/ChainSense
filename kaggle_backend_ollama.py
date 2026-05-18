# ============================================================================
# 🌍 ChainSense Edge — Kaggle Backend (OLLAMA SPECIAL TRACK VERSION)
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
    # 1. Install zstd (Dibutuhkan untuk ekstrak Ollama)
    os.system("apt-get update -y && apt-get install -y zstd")
    
    # 2. Cek apakah ollama ada
    if subprocess.run(["which", "ollama"], capture_output=True).returncode != 0:
        print("📥 Installing Ollama binary...")
        os.system("curl -fsSL https://ollama.com/install.sh | sh")
    
    # 3. Install Python libraries
    os.system("pip install -q ollama fastapi uvicorn pyngrok nest_asyncio torch-geometric")
    print("✅ Installation complete.")

# Jalankan instalasi otomatis
install_dependencies()

import torch
import torch.nn.functional as F
import numpy as np
import nest_asyncio
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Apply nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chainsense-ollama-backend")

# --- START OLLAMA SERVICE ---
def run_ollama_serve():
    print("🚀 Starting Ollama Service...")
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    # Gunakan full path
    subprocess.Popen(["/usr/local/bin/ollama", "serve"])

Thread(target=run_ollama_serve, daemon=True).start()
time.sleep(15) # Beri waktu lebih lama agar server siap

# --- PULL MODEL ---
# Menggunakan Gemma 4 Effective 2B (E2B) sesuai spesifikasi resmi Ollama
MODEL_NAME = "gemma4:e2b" 
print(f"📥 Pulling model {MODEL_NAME} via Ollama... (Size: 7.2GB, This may take a while)")
pull_result = subprocess.run(["/usr/local/bin/ollama", "pull", MODEL_NAME])

if pull_result.returncode != 0:
    print(f"⚠️ Gagal menarik {MODEL_NAME}. Mencoba model alternatif gemma2:2b...")
    MODEL_NAME = "gemma2:2b"
    subprocess.run(["/usr/local/bin/ollama", "pull", MODEL_NAME])

print(f"✅ Model {MODEL_NAME} is ready!")

# ============================================================================
# SECTION 1: GNN ENGINE (PyTorch Geometric)
# ============================================================================
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False

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
            self.model = GCNRiskPredictor().to(self.device)
            self.model.eval()
        else: self.model = None

    def predict(self, node_features, edge_list):
        if not PYGEOMETRIC_AVAILABLE or self.model is None: return self._fallback(node_features)
        try:
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
            with torch.no_grad():
                probs = self.model(x, edge_index)
            scores = probs.cpu().numpy().tolist()
            return {"risk_scores": [round(s, 4) for s in scores], "model_type": "GCN"}
        except: return self._fallback(node_features)

    def _fallback(self, node_features):
        scores = [round(min(max((f[2]*0.6 + f[1]*0.4), 0), 1), 4) for f in node_features]
        return {"risk_scores": scores, "model_type": "heuristic_fallback"}

# ============================================================================
# SECTION 2: FASTAPI & OLLAMA INTEGRATION
# ============================================================================
app = FastAPI(title="ChainSense Ollama Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
gnn_predictor = GNNPredictor()

class UnifiedRequest(BaseModel):
    task: str
    summary: Optional[str] = None
    target_schema: Optional[Dict] = None
    user_columns: Optional[List[str]] = None
    node_features: Optional[List[List[float]]] = None
    edge_list: Optional[List[List[int]]] = None
    node_names: Optional[List[str]] = None

def query_ollama(prompt: str, is_json: bool = False):
    """Inference via Ollama with optional JSON mode and robust parsing"""
    try:
        # Use native JSON mode if requested
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=[{'role': 'user', 'content': prompt}],
            format='json' if is_json else None
        )
        content = response['message']['content']
        
        if is_json:
            # Clean up potential markdown formatting
            import re
            content = re.sub(r'```json\s*|\s*```', '', content).strip()
            # If it still contains non-JSON filler, try to find the first '{' and last '}'
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                content = match.group(1)
        return content
    except Exception as e:
        logger.error(f"Ollama Error: {e}")
        return f"❌ Ollama Error: {str(e)}"

@app.post("/analyze")
async def analyze(req: UnifiedRequest):
    if req.task == "analysis":
        prompt = f"Analisa data supply chain berikut dan berikan instruksi mitigasi darurat: {req.summary}"
        return {"response": query_ollama(prompt)}
    
    elif req.task == "gnn_predict":
        gnn_res = gnn_predictor.predict(req.node_features, req.edge_list)
        xai_prompt = f"Berdasarkan skor risiko GNN ini: {gnn_res['risk_scores'][:10]}, jelaskan kerentanan jaringan secara singkat."
        return {
            "gnn_result": gnn_res,
            "xai_narrative": query_ollama(xai_prompt),
            "high_risk_nodes": [req.node_names[i] for i, s in enumerate(gnn_res['risk_scores']) if s > 0.6]
        }
    
    elif req.task == "mapping":
        prompt = f"""Anda adalah seorang Ahli Teknik Data untuk sistem rantai pasok.
Tugas Anda adalah memetakan nama kolom dari pengguna ke skema standar.

**Skema Target:**
```json
{json.dumps(req.target_schema, indent=2)}
```

**Kolom Pengguna:**
{json.dumps(req.user_columns)}

**Instruksi:**
1.  Analisis makna semantik dari setiap kolom pengguna.
2.  Petakan setiap kolom pengguna ke kunci yang paling sesuai dari **Skema Target**.
3.  **kunci** dari output JSON harus berupa **Kolom Pengguna**.
4.  **nilai** dari output JSON harus berupa kunci yang sesuai dari **Skema Target**.
5.  Hanya sertakan pemetaan yang Anda yakini. Jika kolom pengguna tidak cocok dengan kunci skema apa pun, abaikan.
6.  Hasilkan HANYA objek JSON yang valid. JANGAN sertakan penjelasan, markdown, atau blok kode apa pun.

**Contoh:**
Jika kolom pengguna adalah `["Distributor", "Kota", "Tanggal"]`, output yang benar adalah:
```json
{{
  "Distributor": "Vendor_Name",
  "Kota": "Customer_Location",
  "Tanggal": "Order_Date"
}}
```"""
        raw_json = query_ollama(prompt, is_json=True)
        try:
            return {"mapping": json.loads(raw_json)}
        except Exception as e:
            logger.error(f"JSON Parse Error: {e} | Raw: {raw_json}")
            raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {raw_json}")

# --- SECURE TOKEN MANAGEMENT ---
NGROK_AUTH_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    NGROK_AUTH_TOKEN = user_secrets.get_secret("NGROK_TOKEN")
    print("🔐 Token Ngrok dimuat dari Secrets.")
except Exception:
    import getpass
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
            print(f"\n" + "="*50)
            print(f"🌍 OLLAMA BACKEND LIVE: {public_url}")
            print(f"📋 Salin URL di atas ke BACKEND_URL di app.py Anda")
            print("="*50)
        except Exception as e:
            print(f"❌ Gagal membuka tunnel: {e}")
