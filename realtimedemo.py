"""
realtime_demo.py — Real-time emotion recognition demo
Runs a local HTTP server with a browser UI for:
  • Speech-only  : record mic audio → predict emotion
  • Text-only    : type a sentence  → predict emotion
  • Fusion       : record + type    → predict emotion (both modalities)

Usage:
    python realtime_demo.py
Then open http://localhost:7860 in your browser.
"""

import os, sys, json, io, base64, threading, traceback
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from transformers import DistilBertTokenizer, DistilBertModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_device, set_seed,
    EMOTION_LABELS, NUM_CLASSES, IDX_TO_EMOTION
)
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train   import TextEmotionModel, build_text
from models.fusion_pipeline.train import FusionEmotionModel

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'Results'
SR          = 22050
DURATION    = 4.0
MAX_AUDIO   = 345
MAX_TEXT    = 32
PORT        = 7860

EMOTION_EMOJI = {
    'angry':   '😡', 'disgust': '🤢', 'fear':    '😨',
    'happy':   '😊', 'neutral': '😐', 'ps':      '😲', 'sad': '😢'
}

# ── Key remapping (handles fc1/fc2 legacy checkpoints) ───────────────────────
FC_KEY_MAP = {
    'fc1.weight': 'classifier.1.weight', 'fc1.bias': 'classifier.1.bias',
    'fc2.weight': 'classifier.4.weight', 'fc2.bias': 'classifier.4.bias',
    'text_enc.fc1.weight': 'text_enc.classifier.1.weight',
    'text_enc.fc1.bias':   'text_enc.classifier.1.bias',
    'text_enc.fc2.weight': 'text_enc.classifier.4.weight',
    'text_enc.fc2.bias':   'text_enc.classifier.4.bias',
}

def remap_sd(sd):
    return {FC_KEY_MAP.get(k, k): v for k, v in sd.items()}

def robust_load(model, path, device):
    sd = remap_sd(torch.load(path, map_location=device, weights_only=False))
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ── Audio feature extraction (matches training exactly) ──────────────────────
def wav_to_features(waveform, sr):
    max_len = int(sr * DURATION)
    if len(waveform) < max_len:
        waveform = np.pad(waveform, (0, max_len - len(waveform)))
    else:
        waveform = waveform[:max_len]
    mfcc   = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40,
                                    n_fft=512, hop_length=256)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat   = np.vstack([mfcc, delta, delta2]).T   # (T, 120)
    if feat.shape[0] < MAX_AUDIO:
        feat = np.vstack([feat, np.zeros((MAX_AUDIO - feat.shape[0], 120), dtype=np.float32)])
    else:
        feat = feat[:MAX_AUDIO]
    mean = feat.mean(0, keepdims=True); std = feat.std(0, keepdims=True) + 1e-8
    return torch.tensor((feat - mean) / std, dtype=torch.float32).unsqueeze(0)

# ── Prediction helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def predict_speech(model, device, waveform, sr):
    feat   = wav_to_features(waveform, sr).to(device)
    logits = model(feat)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs.tolist()

@torch.no_grad()
def predict_text(model, tokenizer, device, text):
    enc    = tokenizer(text, max_length=MAX_TEXT, padding='max_length',
                       truncation=True, return_tensors='pt')
    ids    = enc['input_ids'].to(device)
    attn   = enc['attention_mask'].to(device)
    logits = model(ids, attn)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs.tolist()

@torch.no_grad()
def predict_fusion(model, tokenizer, device, waveform, sr, text):
    feat = wav_to_features(waveform, sr).to(device)
    enc  = tokenizer(text, max_length=MAX_TEXT, padding='max_length',
                     truncation=True, return_tensors='pt')
    ids  = enc['input_ids'].to(device)
    attn = enc['attention_mask'].to(device)
    logits = model(feat, ids, attn)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return int(probs.argmax()), probs.tolist()

# ── Globals populated at startup ──────────────────────────────────────────────
device    = None
tokenizer = None
sp_model  = None
tx_model  = None
fu_model  = None

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # silence default access log

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_html(HTML)

    def do_POST(self):
        try:
            length  = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length))
            path    = urlparse(self.path).path

            if path == '/predict/speech':
                waveform, sr = self._decode_audio(payload['audio'])
                idx, probs   = predict_speech(sp_model, device, waveform, sr)
                self.send_json(200, self._fmt(idx, probs))

            elif path == '/predict/text':
                text       = payload['text'].strip()
                idx, probs = predict_text(tx_model, tokenizer, device, text)
                self.send_json(200, self._fmt(idx, probs))

            elif path == '/predict/fusion':
                waveform, sr = self._decode_audio(payload['audio'])
                text         = payload['text'].strip()
                idx, probs   = predict_fusion(fu_model, tokenizer, device,
                                               waveform, sr, text)
                self.send_json(200, self._fmt(idx, probs))

            else:
                self.send_json(404, {'error': 'unknown endpoint'})

        except Exception as e:
            self.send_json(500, {'error': str(e), 'trace': traceback.format_exc()})

    # ── helpers ──────────────────────────────────────────────────────────────
    def _decode_audio(self, b64):
        raw      = base64.b64decode(b64)
        buf      = io.BytesIO(raw)
        waveform, sr = sf.read(buf, dtype='float32')
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR)
            sr = SR
        return waveform, sr

    def _fmt(self, idx, probs):
        label = EMOTION_LABELS[idx]
        return {
            'emotion':   label,
            'emoji':     EMOTION_EMOJI[label],
            'confidence': round(float(probs[idx]) * 100, 1),
            'probs':     {EMOTION_LABELS[i]: round(float(p)*100, 1)
                          for i, p in enumerate(probs)}
        }

# ── HTML UI (single-file, no external CDN needed) ────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Emotion Recognition · Live Demo</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:      #0b0e14;
    --surface: #12161f;
    --card:    #1a1f2e;
    --border:  #252b3b;
    --accent:  #6ee7b7;
    --accent2: #818cf8;
    --accent3: #f472b6;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --angry:   #ef4444; --disgust:#a3e635; --fear:#f97316;
    --happy:   #facc15; --neutral:#94a3b8; --ps:#c084fc; --sad:#60a5fa;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* ── Header ── */
  header {
    width: 100%;
    padding: 28px 40px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 14px;
  }
  header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    letter-spacing: -0.5px;
    color: var(--accent);
  }
  header span {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  /* ── Main grid ── */
  main {
    width: 100%;
    max-width: 1100px;
    padding: 36px 24px 60px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
  }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 26px 22px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    position: relative;
    overflow: hidden;
    transition: border-color .25s;
  }
  .card:hover { border-color: #334155; }
  .card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    opacity: .06;
    pointer-events: none;
  }
  .card.speech::before { background: var(--accent); }
  .card.text::before   { background: var(--accent2); }
  .card.fusion::before { background: var(--accent3); }

  .card-title {
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card.speech .card-title { color: var(--accent); }
  .card.text   .card-title { color: var(--accent2); }
  .card.fusion .card-title { color: var(--accent3); }

  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  .card.speech .dot { background: var(--accent); }
  .card.text   .dot { background: var(--accent2); }
  .card.fusion .dot { background: var(--accent3); }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.4; transform:scale(1.4); }
  }

  /* ── Recorder ── */
  .recorder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }

  .rec-btn {
    width: 68px; height: 68px;
    border-radius: 50%;
    border: 2px solid currentColor;
    background: transparent;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    transition: all .2s;
    position: relative;
  }
  .rec-btn:hover { transform: scale(1.08); }
  .rec-btn.recording {
    background: #ef444422;
    border-color: #ef4444;
    animation: ring 1.2s infinite;
  }
  @keyframes ring {
    0%,100% { box-shadow: 0 0 0 0 #ef444455; }
    50%      { box-shadow: 0 0 0 12px #ef444400; }
  }

  .rec-status {
    font-size: .78rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    min-height: 18px;
  }
  .rec-status.active { color: #ef4444; }

  /* ── Waveform canvas ── */
  canvas.waveform {
    width: 100%;
    height: 48px;
    border-radius: 8px;
    background: var(--surface);
    display: none;
  }

  /* ── Text input ── */
  textarea {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: .9rem;
    padding: 12px 14px;
    resize: vertical;
    min-height: 80px;
    outline: none;
    transition: border-color .2s;
  }
  textarea:focus { border-color: #475569; }

  /* ── Predict button ── */
  .predict-btn {
    width: 100%;
    padding: 11px;
    border-radius: 10px;
    border: none;
    font-family: 'Space Mono', monospace;
    font-size: .78rem;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all .2s;
    font-weight: 700;
  }
  .card.speech .predict-btn { background: var(--accent);  color: #0b0e14; }
  .card.text   .predict-btn { background: var(--accent2); color: #fff; }
  .card.fusion .predict-btn { background: var(--accent3); color: #0b0e14; }
  .predict-btn:hover { filter: brightness(1.12); transform: translateY(-1px); }
  .predict-btn:disabled { opacity: .45; cursor: not-allowed; transform: none; }

  /* ── Result box ── */
  .result {
    background: var(--surface);
    border-radius: 12px;
    padding: 16px;
    display: none;
    flex-direction: column;
    gap: 12px;
    animation: fadeIn .3s ease;
  }
  .result.show { display: flex; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }

  .result-top {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .result-emoji { font-size: 2.4rem; line-height: 1; }
  .result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .result-conf {
    font-size: .78rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  /* ── Bar chart ── */
  .bars { display: flex; flex-direction: column; gap: 5px; }
  .bar-row {
    display: grid;
    grid-template-columns: 56px 1fr 36px;
    align-items: center;
    gap: 7px;
    font-size: .72rem;
    font-family: 'Space Mono', monospace;
  }
  .bar-label { color: var(--muted); text-align: right; }
  .bar-track {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .5s cubic-bezier(.4,0,.2,1);
  }
  .bar-pct { color: var(--muted); text-align: right; }

  /* ── Spinner ── */
  .spinner {
    width: 20px; height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    margin: 0 auto;
    display: none;
  }
  .spinner.show { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-msg {
    font-size: .78rem;
    color: #ef4444;
    font-family: 'Space Mono', monospace;
    display: none;
    padding: 8px 10px;
    background: #ef444411;
    border-radius: 8px;
  }
  .error-msg.show { display: block; }

  /* ── Footer ── */
  footer {
    color: var(--muted);
    font-size: .72rem;
    font-family: 'Space Mono', monospace;
    padding: 20px;
    text-align: center;
    border-top: 1px solid var(--border);
    width: 100%;
  }

  @media (max-width: 760px) {
    main { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <h1>EMOTION · RECOGNITION</h1>
  <span>real-time demo · speech + text + fusion</span>
</header>

<main>

  <!-- ── SPEECH CARD ─────────────────────────────────────── -->
  <div class="card speech">
    <div class="card-title"><span class="dot"></span>Speech Pipeline</div>

    <div class="recorder">
      <button class="rec-btn" id="sp-rec-btn" style="color:var(--accent)"
              title="Hold to record">🎤</button>
      <div class="rec-status" id="sp-status">click to record (4 s)</div>
    </div>

    <canvas class="waveform" id="sp-wave"></canvas>

    <button class="predict-btn" id="sp-predict" disabled>
      PREDICT EMOTION
    </button>

    <div class="spinner" id="sp-spin"></div>
    <div class="error-msg" id="sp-err"></div>
    <div class="result" id="sp-result"></div>
  </div>

  <!-- ── TEXT CARD ───────────────────────────────────────── -->
  <div class="card text">
    <div class="card-title"><span class="dot"></span>Text Pipeline</div>

    <textarea id="tx-input"
              placeholder="Type how you feel, or paste a sentence…&#10;e.g. I am so happy and excited today!"></textarea>

    <button class="predict-btn" id="tx-predict">PREDICT EMOTION</button>

    <div class="spinner" id="tx-spin"></div>
    <div class="error-msg" id="tx-err"></div>
    <div class="result" id="tx-result"></div>
  </div>

  <!-- ── FUSION CARD ─────────────────────────────────────── -->
  <div class="card fusion">
    <div class="card-title"><span class="dot"></span>Fusion Pipeline</div>

    <div class="recorder">
      <button class="rec-btn" id="fu-rec-btn" style="color:var(--accent3)"
              title="Record audio">🎙️</button>
      <div class="rec-status" id="fu-status">click to record (4 s)</div>
    </div>

    <canvas class="waveform" id="fu-wave"></canvas>

    <textarea id="fu-input"
              placeholder="Also type what you said (or how you feel)…"></textarea>

    <button class="predict-btn" id="fu-predict" disabled>
      PREDICT EMOTION
    </button>

    <div class="spinner" id="fu-spin"></div>
    <div class="error-msg" id="fu-err"></div>
    <div class="result" id="fu-result"></div>
  </div>

</main>

<footer>models: BiLSTM speech · DistilBERT text · gated fusion &nbsp;|&nbsp; TESS dataset · 7 emotions</footer>

<script>
// ── Emotion colours ───────────────────────────────────────────────────────────
const EMO_COLOR = {
  angry:'#ef4444', disgust:'#a3e635', fear:'#f97316',
  happy:'#facc15', neutral:'#94a3b8', ps:'#c084fc', sad:'#60a5fa'
};
const EMO_ORDER = ['angry','disgust','fear','happy','neutral','ps','sad'];

// ── Audio helpers ─────────────────────────────────────────────────────────────
async function encodeWav(buffer, sampleRate) {
  // Encode raw Float32 PCM to 16-bit WAV, return base64
  const ch = buffer.numberOfChannels;
  const len = buffer.length;
  const data = buffer.getChannelData(0);
  const wavBuf = new ArrayBuffer(44 + len * 2);
  const view = new DataView(wavBuf);
  const write = (o, s) => { for (let i=0;i<s.length;i++) view.setUint8(o+i, s.charCodeAt(i)); };
  write(0,'RIFF'); view.setUint32(4, 36+len*2, true);
  write(8,'WAVE'); write(12,'fmt '); view.setUint32(16,16,true);
  view.setUint16(20,1,true); view.setUint16(22,1,true);
  view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*2,true);
  view.setUint16(32,2,true); view.setUint16(34,16,true);
  write(36,'data'); view.setUint32(40,len*2,true);
  for (let i=0;i<len;i++) {
    const s = Math.max(-1,Math.min(1,data[i]));
    view.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
  }
  return btoa(String.fromCharCode(...new Uint8Array(wavBuf)));
}

// Draw simple waveform on canvas
function drawWave(canvas, buffer) {
  canvas.style.display = 'block';
  const ctx  = canvas.getContext('2d');
  const data = buffer.getChannelData(0);
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = 48;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#12161f';
  ctx.fillRect(0,0,W,H);
  const step = Math.floor(data.length / W);
  ctx.strokeStyle = '#6ee7b7';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let x=0; x<W; x++) {
    const s = data[x * step];
    const y = (1 - s) / 2 * H;
    x===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  }
  ctx.stroke();
}

// ── Recorder factory ──────────────────────────────────────────────────────────
function makeRecorder(btnId, statusId, waveId, predictBtnId, onReady) {
  const btn      = document.getElementById(btnId);
  const status   = document.getElementById(statusId);
  const waveEl   = document.getElementById(waveId);
  const predBtn  = document.getElementById(predictBtnId);
  let mediaRec, chunks = [], audioCtx, audioBuffer;

  btn.addEventListener('click', async () => {
    if (mediaRec && mediaRec.state === 'recording') return;

    // request mic
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch(e) {
      status.textContent = '⚠ mic access denied';
      return;
    }

    chunks = [];
    mediaRec = new MediaRecorder(stream);
    mediaRec.ondataavailable = e => chunks.push(e.data);
    mediaRec.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const ab   = await blob.arrayBuffer();
      audioCtx   = new AudioContext();
      audioBuffer = await audioCtx.decodeAudioData(ab);
      drawWave(waveEl, audioBuffer);
      status.textContent = `✓ recorded ${audioBuffer.duration.toFixed(1)}s`;
      status.classList.remove('active');
      predBtn.disabled = false;
      onReady && onReady(audioBuffer, audioCtx.sampleRate);
      stream.getTracks().forEach(t => t.stop());
    };

    mediaRec.start();
    btn.classList.add('recording');
    btn.textContent = '⏹';
    status.textContent = 'recording…';
    status.classList.add('active');

    // auto-stop at 4 s
    setTimeout(() => {
      if (mediaRec.state === 'recording') {
        mediaRec.stop();
        btn.classList.remove('recording');
        btn.textContent = btnId.startsWith('fu') ? '🎙️' : '🎤';
      }
    }, 4000);
  });

  return {
    getBase64: async () => {
      if (!audioBuffer) return null;
      return encodeWav(audioBuffer, audioCtx.sampleRate);
    }
  };
}

// ── Result renderer ───────────────────────────────────────────────────────────
function renderResult(containerId, data) {
  const el = document.getElementById(containerId);
  const color = EMO_COLOR[data.emotion] || '#6ee7b7';
  const bars = EMO_ORDER.map(e => {
    const pct = data.probs[e] || 0;
    return `
      <div class="bar-row">
        <div class="bar-label">${e}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%;background:${EMO_COLOR[e]}"></div>
        </div>
        <div class="bar-pct">${pct.toFixed(0)}%</div>
      </div>`;
  }).join('');

  el.innerHTML = `
    <div class="result-top">
      <div class="result-emoji">${data.emoji}</div>
      <div>
        <div class="result-label" style="color:${color}">${data.emotion}</div>
        <div class="result-conf">${data.confidence}% confidence</div>
      </div>
    </div>
    <div class="bars">${bars}</div>`;
  el.classList.add('show');
}

// ── Generic predict call ──────────────────────────────────────────────────────
async function callPredict(endpoint, payload, spinId, errId, resultId, btnId) {
  const spin = document.getElementById(spinId);
  const err  = document.getElementById(errId);
  const btn  = document.getElementById(btnId);

  spin.classList.add('show');
  err.classList.remove('show');
  btn.disabled = true;

  try {
    const res  = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'server error');
    renderResult(resultId, data);
  } catch(e) {
    err.textContent = '⚠ ' + e.message;
    err.classList.add('show');
  } finally {
    spin.classList.remove('show');
    btn.disabled = false;
  }
}

// ── SPEECH setup ──────────────────────────────────────────────────────────────
const spRec = makeRecorder('sp-rec-btn','sp-status','sp-wave','sp-predict');
document.getElementById('sp-predict').addEventListener('click', async () => {
  const b64 = await spRec.getBase64();
  if (!b64) return;
  callPredict('/predict/speech', {audio:b64}, 'sp-spin','sp-err','sp-result','sp-predict');
});

// ── TEXT setup ────────────────────────────────────────────────────────────────
document.getElementById('tx-predict').addEventListener('click', () => {
  const text = document.getElementById('tx-input').value.trim();
  if (!text) { document.getElementById('tx-err').textContent='⚠ please enter some text'; document.getElementById('tx-err').classList.add('show'); return; }
  callPredict('/predict/text', {text}, 'tx-spin','tx-err','tx-result','tx-predict');
});
document.getElementById('tx-input').addEventListener('keydown', e => {
  if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('tx-predict').click(); }
});

// ── FUSION setup ──────────────────────────────────────────────────────────────
const fuRec = makeRecorder('fu-rec-btn','fu-status','fu-wave','fu-predict');
document.getElementById('fu-predict').addEventListener('click', async () => {
  const b64  = await fuRec.getBase64();
  const text = document.getElementById('fu-input').value.trim();
  if (!b64)  { document.getElementById('fu-err').textContent='⚠ record audio first'; document.getElementById('fu-err').classList.add('show'); return; }
  if (!text) { document.getElementById('fu-err').textContent='⚠ also type some text'; document.getElementById('fu-err').classList.add('show'); return; }
  callPredict('/predict/fusion',{audio:b64,text},'fu-spin','fu-err','fu-result','fu-predict');
});
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
def load_models():
    global device, tokenizer, sp_model, tx_model, fu_model
    set_seed(42)
    device    = get_device()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    print(f'  Loading speech model …')
    sp_model = SpeechEmotionModel().to(device)
    robust_load(sp_model, os.path.join(RESULTS_DIR, 'speech_best_model.pt'), device)

    print(f'  Loading text model …')
    tx_model = TextEmotionModel().to(device)
    robust_load(tx_model, os.path.join(RESULTS_DIR, 'text_best_model.pt'), device)

    print(f'  Loading fusion model …')
    fu_model = FusionEmotionModel().to(device)
    robust_load(fu_model, os.path.join(RESULTS_DIR, 'fusion_best_model.pt'), device)

    print(f'  All models loaded on {device}')


if __name__ == '__main__':
    print('\n' + '━'*52)
    print('  EMOTION RECOGNITION — Real-time Demo')
    print('━'*52)
    load_models()
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'\n  Server running at  http://localhost:{PORT}')
    print(f'  Open that URL in your browser.')
    print(f'  Press Ctrl-C to stop.\n')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n  Stopped.')