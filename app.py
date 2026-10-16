import os
import json
import time
import random
import wave
import uuid
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote

import gradio as gr
import numpy as np
import pretty_midi
import tensorflow as tf
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# ============================================================
# HATHOR AI — CINEMATIC MODEL + SECURE LOGIN CONFIGURATION
# ============================================================
BASE = Path(__file__).resolve().parent
CINEMATIC_DIR = BASE / "cinematic-model"
MODEL_PATH = CINEMATIC_DIR / "checkpoints" / "best_model.keras"
DATA_PATH = CINEMATIC_DIR / "processed" / "dataset_cache.npz"
VOCAB_PATH = CINEMATIC_DIR / "processed" / "vocabulary.json"
DURATION_PATH = CINEMATIC_DIR / "export_package" / "duration_bins.json"

OUTPUT_DIR = BASE / "outputs" / "midi"
AUDIO_DIR = BASE / "outputs" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_CLIENT_ID = os.getenv(
    "GOOGLE_CLIENT_ID",
    "426733701357-loodtt4p0fk9lmhf7qtsq44a9the3sm4.apps.googleusercontent.com",
).strip()
HATHOR_LOG_URL = os.getenv(
    "HATHOR_LOG_URL",
    "https://script.google.com/macros/s/AKfycbyWBLUomdvCGk7PGYGMNuF2IoM0YqHifi-0-GqdMn9lYmXt2OvyEymX62se4sIMXLce/exec",
).strip()
HATHOR_LOG_SECRET = os.getenv("HATHOR_LOG_SECRET", "").strip()
SESSION_SECRET = os.getenv("HATHOR_SESSION_SECRET", "").strip()

# Do not silently run an insecure session configuration in production.
if not SESSION_SECRET:
    # A per-process fallback keeps local testing possible, but Render should
    # have HATHOR_SESSION_SECRET set to a long random value.
    SESSION_SECRET = os.urandom(32).hex()
    print("WARNING: HATHOR_SESSION_SECRET is not set; sessions reset on restart.")
if not HATHOR_LOG_SECRET:
    print("WARNING: HATHOR_LOG_SECRET is not set; spreadsheet login tracking will fail until configured.")

for required_path in (MODEL_PATH, DATA_PATH, VOCAB_PATH, DURATION_PATH):
    if not required_path.exists():
        raise FileNotFoundError(f"Required cinematic-model file is missing: {required_path}")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    token_to_id = json.load(f)

# This model's vocabulary maps strings such as "60,2" -> token ID.
id_to_token = {int(token_id): token for token, token_id in token_to_id.items()}
with open(DURATION_PATH, "r", encoding="utf-8") as f:
    DURATION_BINS = [float(v) for v in json.load(f)]

# Load only the validation seed bank, not the very large training matrix.
# X_val is enough to seed generation and uses substantially less memory.
with np.load(DATA_PATH, allow_pickle=False) as data:
    SEED_BANK = data["X_val"]

SEQUENCE_LENGTH = int(SEED_BANK.shape[1])
if SEQUENCE_LENGTH != 64:
    raise ValueError(f"Expected the cinematic model's 64-token context, got {SEQUENCE_LENGTH}.")
if len(token_to_id) != 519:
    raise ValueError(f"Expected 519 cinematic tokens, got {len(token_to_id)}.")
if len(DURATION_BINS) != 6:
    raise ValueError(f"Expected 6 cinematic duration bins, got {len(DURATION_BINS)}.")

# Load the NEW cinematic checkpoint (not models/hathor_best.keras).
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
output_size = int(model.output_shape[-1])
if output_size != len(token_to_id):
    raise ValueError(
        f"Cinematic model output has {output_size} classes but vocabulary has {len(token_to_id)} tokens."
    )

# ============================================================
# LOGIN TRACKING + SESSION HELPERS
# ============================================================
def log_successful_login(method: str, user: dict) -> None:
    """Log verified successful sign-ins to the existing Apps Script endpoint."""
    if not HATHOR_LOG_SECRET:
        raise RuntimeError("Login tracking is not configured. Add HATHOR_LOG_SECRET to Render environment variables.")
    payload = {
        "secret": HATHOR_LOG_SECRET,
        "login_method": method,
        "name": str(user.get("name") or ("Guest" if method == "Guest" else "Google user"))[:160],
        "email": str(user.get("email") or "")[:254],
        "session_id": str(user.get("session_id") or "")[:100],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_agent": str(user.get("user_agent") or "")[:300],
    }
    response = requests.post(HATHOR_LOG_URL, json=payload, timeout=12, allow_redirects=True)
    response.raise_for_status()
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("The login tracker did not return JSON; verify the Apps Script deployment.") from exc
    # Require the expected success acknowledgement so a redirect/HTML error page
    # cannot be mistaken for a saved spreadsheet record.
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, dict) else None
        raise RuntimeError(str(detail or "Spreadsheet did not confirm saving the login record."))


def get_current_user(request: Request):
    user = request.session.get("hathor_user")
    if not isinstance(user, dict) or not user.get("session_id"):
        return None
    return user


def gradio_auth_dependency(request: Request):
    """Gradio calls this for mounted app requests; unauthenticated users are denied."""
    user = get_current_user(request)
    return user.get("email") or user.get("name") or "Guest" if user else None

# ============================================================
# CINEMATIC TOKEN SAMPLING
# ============================================================
def sample_next_token(probabilities, temperature: float) -> int:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    temperature = float(np.clip(temperature, 0.25, 1.5))
    probs = np.maximum(probs, 1e-9)
    logits = np.log(probs) / temperature
    logits -= np.max(logits)
    scaled = np.exp(logits)
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(probs))
    scaled /= total
    return int(np.random.choice(len(scaled), p=scaled))


def generate_tokens(length: int, temperature: float):
    seed_index = random.randrange(len(SEED_BANK))
    window = [int(v) for v in SEED_BANK[seed_index][-SEQUENCE_LENGTH:]]
    generated = []
    for _ in range(int(length)):
        inp = np.asarray([window], dtype=np.int32)
        prediction = model(inp, training=False).numpy()[0]
        next_id = sample_next_token(prediction, temperature)
        if next_id not in id_to_token:
            next_id = int(np.argmax(prediction))
        generated.append(next_id)
        window = window[1:] + [next_id]
    decoded = []
    for token_id in generated:
        try:
            pitch_text, duration_text = id_to_token[int(token_id)].split(",", 1)
            pitch, duration_idx = int(pitch_text), int(duration_text)
            if not 0 <= duration_idx < len(DURATION_BINS):
                continue
            decoded.append((pitch, duration_idx))
        except (KeyError, ValueError, TypeError):
            continue
    return decoded


def duration_seconds(duration_idx: int, tempo: float) -> float:
    if not 0 <= int(duration_idx) < len(DURATION_BINS):
        raise ValueError("The cinematic model generated an invalid duration class.")
    # Duration bins are stored as beat-like values, with the final class capped
    # at four beats. Convert them to real seconds at the chosen BPM.
    beats = float(np.clip(DURATION_BINS[int(duration_idx)], 0.08, 4.0))
    return max(0.04, beats * 60.0 / float(tempo))


def clean_notes(notes):
    cleaned = []
    for pitch, duration_idx in notes:
        pitch = int(np.clip(int(pitch), 21, 108))
        duration_idx = int(duration_idx)
        if 0 <= duration_idx < len(DURATION_BINS):
            cleaned.append((pitch, duration_idx))
    return cleaned

# ============================================================
# MIDI EXPORT + WAV PREVIEW
# ============================================================
def create_midi(notes, tempo, instrument_name):
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)
    current_time = 0.0
    for pitch, duration_idx in notes:
        duration = duration_seconds(duration_idx, tempo)
        note = pretty_midi.Note(
            velocity=92,
            pitch=int(np.clip(pitch, 0, 127)),
            start=current_time,
            end=current_time + max(duration, 0.05),
        )
        instrument.notes.append(note)
        current_time += max(duration, 0.05)
    midi.instruments.append(instrument)
    stamp = int(time.time())
    output_path = OUTPUT_DIR / f"hathor_cinematic_{stamp}_{random.randrange(100000):05d}.mid"
    midi.write(str(output_path))
    check = pretty_midi.PrettyMIDI(str(output_path))
    if not check.instruments or not check.instruments[0].notes:
        raise RuntimeError("The exported MIDI file could not be validated.")
    return midi, str(output_path)


def instrument_waveform(phase, instrument_name):
    name = instrument_name.lower()
    if "flute" in name:
        return (np.sin(phase) + 0.12 * np.sin(2 * phase)) / 1.12
    if any(word in name for word in ("violin", "cello", "strings")):
        return (np.sin(phase) + .45*np.sin(2*phase) + .22*np.sin(3*phase) + .1*np.sin(4*phase)) / 1.77
    if "guitar" in name:
        return (np.sin(phase) + .35*np.sin(2*phase) + .18*np.sin(3*phase)) / 1.53
    if "trumpet" in name or "sax" in name:
        return (np.sin(phase) + .55*np.sin(2*phase) + .35*np.sin(3*phase) + .2*np.sin(4*phase)) / 2.10
    if "organ" in name:
        return (np.sin(phase) + .5*np.sin(2*phase) + .25*np.sin(3*phase)) / 1.75
    return (np.sin(phase) + .3*np.sin(2*phase) + .12*np.sin(3*phase)) / 1.42


def make_audio_preview(notes, tempo, instrument_name, audio_path):
    sample_rate = 22050
    events = []
    current_time = 0.0
    for pitch, duration_idx in notes:
        duration = duration_seconds(duration_idx, tempo)
        events.append((int(np.clip(pitch, 0, 127)), current_time, duration))
        current_time += duration
    total_seconds = min(current_time + 0.4, 120.0)
    audio = np.zeros(max(1, int(total_seconds * sample_rate)), dtype=np.float64)
    for pitch, start_time, duration in events:
        if start_time >= total_seconds:
            break
        start = int(start_time * sample_rate)
        if start >= len(audio):
            break
        count = min(max(1, int(duration * sample_rate)), len(audio) - start)
        if count <= 2:
            continue
        t = np.arange(count, dtype=np.float64) / sample_rate
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        tone = instrument_waveform(2.0 * np.pi * frequency * t, instrument_name)
        attack = min(int(.012 * sample_rate), count // 3)
        release = min(int(.07 * sample_rate), count // 3)
        envelope = np.exp(-3.0 * t / max(duration, .1))
        if attack > 1:
            envelope[:attack] *= np.linspace(0.0, 1.0, attack)
        if release > 1:
            envelope[-release:] *= np.linspace(1.0, 0.0, release)
        audio[start:start+count] += tone * envelope * .22
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= min(1.0, .9 / peak)
    pcm = (audio * 32767).astype(np.int16)
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return str(audio_path)


def generate_music(length, temperature, tempo, instrument_name, history_items):
    try:
        notes = clean_notes(generate_tokens(int(length), float(temperature)))
        if not notes:
            raise ValueError("The cinematic model did not produce valid notes.")
        _, midi_path = create_midi(notes, float(tempo), instrument_name)
        audio_path = AUDIO_DIR / (Path(midi_path).stem + ".wav")
        make_audio_preview(notes, float(tempo), instrument_name, audio_path)
        label = f"{datetime.now().strftime('%H:%M:%S')} · {instrument_name} · {len(notes)} notes"
        history_items = list(history_items or [])
        history_items.insert(0, {
            "label": label,
            "midi": midi_path,
            "audio": str(audio_path),
            "status": f"Cinematic composition ready · {len(notes)} notes · {int(tempo)} BPM · {instrument_name}",
        })
        history_items = history_items[:10]
        return (
            midi_path,
            str(audio_path),
            f"CINEMATIC MODEL ONLINE\nInstrument: {instrument_name}\nNotes: {len(notes)}\nTempo: {int(tempo)} BPM\nModel: best_model.keras",
            history_items,
            gr.update(choices=[item["label"] for item in history_items], value=label),
        )
    except Exception as exc:
        raise gr.Error(f"Cinematic generation failed: {exc}") from exc


def load_history(label, history_items):
    for item in history_items or []:
        if item["label"] == label:
            return item["midi"], item["audio"], item["status"]
    return None, None, "Choose a composition from your generation history."

# ============================================================
# CYBERPUNK GRADIO STUDIO
# ============================================================
CYBERPUNK_CSS = r"""
:root { color-scheme: dark; }
.gradio-container { max-width: 1180px !important; margin:auto !important; background:#060812 !important; color:#eff7ff !important; font-family:Inter,ui-sans-serif,system-ui,sans-serif !important; }
body,.main { background:#060812 !important; }
footer { display:none !important; }
#hathor-header { position:relative; overflow:hidden; padding:34px 24px 30px; margin:8px 0 22px; border:1px solid #203a55; border-radius:24px; text-align:center; background:radial-gradient(ellipse at 50% -20%,rgba(0,238,255,.22),transparent 60%),radial-gradient(ellipse at 90% 100%,rgba(164,72,255,.15),transparent 45%),#090e1b; box-shadow:0 0 35px rgba(0,238,255,.06),inset 0 0 30px rgba(0,238,255,.025); }
#hathor-brand { color:#55f3ff; font-size:11px; font-weight:900; letter-spacing:5px; margin-bottom:16px; }
#hathor-header h1 { color:#f4f8ff; font-size:clamp(35px,6vw,62px); line-height:1.05; letter-spacing:-2px; margin:0 0 14px; }
#hathor-header h1 span { color:#62f5ff; text-shadow:0 0 24px rgba(98,245,255,.4); }
#hathor-header p { max-width:650px; margin:auto; color:#9aacc9; font-size:14px; line-height:1.85; }
#hathor-wave { color:#60f5ff; letter-spacing:5px; font-size:22px; margin-top:20px; white-space:nowrap; overflow:hidden; text-shadow:0 0 12px rgba(96,245,255,.45); }
.hathor-panel { background:linear-gradient(145deg,rgba(16,25,45,.98),rgba(10,15,29,.98)) !important; border:1px solid #243852 !important; border-radius:19px !important; padding:21px !important; box-shadow:0 8px 30px rgba(0,0,0,.16); }
.hathor-heading { color:#f1f7ff; font-size:20px; font-weight:800; margin-bottom:8px; }
.hathor-description { color:#94a8c4; font-size:13px; line-height:1.7; margin-bottom:19px; }
.hathor-footer { text-align:center; color:#7286a4; font-size:11px; letter-spacing:1px; line-height:2; padding:24px 8px 12px; }
#generate-btn { background:linear-gradient(90deg,#55f3ff,#a46bff) !important; color:#050914 !important; border:0 !important; border-radius:12px !important; min-height:54px !important; font-size:15px !important; font-weight:900 !important; box-shadow:0 0 22px rgba(85,243,255,.16); }
button,input,textarea { border-radius:10px !important; }
label,.label-wrap span { color:#dce8fb !important; }
input[type=range] { accent-color:#55f3ff !important; }
#result-file,#audio-preview { border:1px dashed #385779 !important; border-radius:14px !important; background:#080f1e !important; }
#status-box textarea { color:#b8c9e4 !important; background:#080f1e !important; border:1px solid #243852 !important; }
@media(max-width:640px) { .gradio-container{padding-left:11px!important;padding-right:11px!important} #hathor-header{padding:27px 15px;border-radius:18px} #hathor-header h1{font-size:38px;letter-spacing:-1.2px} #hathor-header p{font-size:13px} .hathor-panel{padding:16px!important} }
"""

with gr.Blocks(
    title="Hathor AI | Cinematic Neural Studio",
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="purple", neutral_hue="slate"),
    css=CYBERPUNK_CSS,
) as demo:
    gr.HTML("""
    <header id="hathor-header">
      <div id="hathor-brand">HΛTHOR AI · CINEMATIC NEURAL STUDIO</div>
      <h1>Compose beyond <span>reality.</span></h1>
      <p>Turn learned musical patterns into cinematic ideas. Shape the sequence, choose a playback voice, preview the result, and export your composition as MIDI.</p>
      <div id="hathor-wave" aria-hidden="true">▁ ▃ ▅ ▂ ▇ ▄ ▆ ▂ ▅ ▇ ▃ ▆ ▂ ▄ ▇ ▃ ▁</div>
    </header>
    """)
    history = gr.State([])
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["hathor-panel"]):
                gr.HTML('<div class="hathor-heading">✦ Composition controls</div><div class="hathor-description">Build a new idea with the cinematic checkpoint. Higher creativity allows more surprising note choices.</div>')
                length = gr.Slider(32, 240, value=120, step=8, label="Sequence length", info="Number of new musical events")
                temperature = gr.Slider(.3, 1.5, value=.85, step=.05, label="Creativity / temperature", info="Lower is more predictable; higher is more adventurous")
                tempo = gr.Slider(60, 180, value=100, step=5, label="Tempo (BPM)")
                instrument = gr.Dropdown(
                    choices=["Acoustic Grand Piano", "Violin", "Cello", "Flute", "Acoustic Guitar (nylon)", "Electric Guitar (clean)", "Trumpet", "Soprano Sax", "Church Organ", "Synth Strings 1"],
                    value="Acoustic Grand Piano", label="Preview instrument",
                )
                generate = gr.Button("✦  GENERATE CINEMATIC MUSIC", variant="primary", elem_id="generate-btn")
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["hathor-panel"]):
                gr.HTML('<div class="hathor-heading">◈ Your composition</div><div class="hathor-description">Preview the generated idea, then download the MIDI file for a DAW or compatible music app.</div>')
                audio_result = gr.Audio(label="WAV preview", type="filepath", interactive=False, elem_id="audio-preview")
                result = gr.File(label="Download MIDI", elem_id="result-file", interactive=False)
                status = gr.Textbox(label="Studio status", value="Cinematic model ready. Create your first composition.", lines=5, interactive=False, elem_id="status-box")
                gr.Markdown("**EXPORT** · MIDI + WAV preview  \n**HISTORY** · Latest 10 compositions in this browser session")
    with gr.Group(elem_classes=["hathor-panel"]):
        gr.HTML('<div class="hathor-heading">◷ Recent compositions</div><div class="hathor-description">Select a previous composition from this session to restore its MIDI and audio preview.</div>')
        history_picker = gr.Dropdown(choices=[], label="Composition history", interactive=True)
    gr.HTML('<div class="hathor-footer">HATHOR AI · CINEMATIC MODEL EDITION<br>FROM NEURAL PATTERNS TO MUSICAL IDEAS</div>')
    generate.click(generate_music, inputs=[length, temperature, tempo, instrument, history], outputs=[result, audio_result, status, history, history_picker])
    history_picker.change(load_history, inputs=[history_picker, history], outputs=[result, audio_result, status])

# ============================================================
# FASTAPI LOGIN GATE — GOOGLE + GUEST + SHEET LOGGING
# ============================================================
app = FastAPI(title="Hathor AI")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="hathor_session",
    max_age=60 * 60 * 24 * 14,
    same_site="lax",
    https_only=os.getenv("RENDER", "").lower() == "true" or bool(os.getenv("RENDER_EXTERNAL_URL")),
)

LOGIN_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#070914"><title>Hathor AI · Access Studio</title>
<script src="https://accounts.google.com/gsi/client" async defer></script>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#060812;color:#eff6ff;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;display:grid;place-items:center;padding:22px;overflow-x:hidden}.bg{position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 12% 15%,rgba(0,238,255,.13),transparent 36%),radial-gradient(ellipse at 90% 85%,rgba(158,72,255,.15),transparent 38%),linear-gradient(rgba(70,130,170,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(70,130,170,.035) 1px,transparent 1px);background-size:auto,auto,36px 36px,36px 36px}.shell{width:min(100%,460px);position:relative}.top{display:flex;align-items:center;justify-content:center;gap:10px;color:#60f4ff;font-size:11px;letter-spacing:4px;font-weight:900;margin-bottom:22px}.orb{width:34px;height:34px;border:1px solid #50e9fa;border-radius:11px;display:grid;place-items:center;box-shadow:0 0 22px #1b879c55;font-size:19px}.card{background:linear-gradient(145deg,rgba(16,25,45,.97),rgba(8,13,27,.98));border:1px solid #243b57;border-radius:25px;padding:34px 29px;box-shadow:0 20px 80px #0007,0 0 45px #00eaff0a}.eyebrow{text-align:center;color:#8fa5c4;font-size:10px;letter-spacing:3px;font-weight:800}.title{text-align:center;font-size:39px;line-height:1.08;letter-spacing:-1.8px;margin:13px 0 13px}.title span{color:#5ff4ff;text-shadow:0 0 25px #5ff4ff44}.sub{text-align:center;color:#9aacc8;font-size:13px;line-height:1.8;margin:0 auto 27px;max-width:330px}.btn{width:100%;min-height:53px;border-radius:13px;border:1px solid #304660;background:#0c1527;color:#f1f6ff;font-size:14px;font-weight:750;display:flex;align-items:center;justify-content:center;gap:12px;cursor:pointer;transition:.18s ease;margin-top:12px}.btn:hover{transform:translateY(-1px);border-color:#60f4ff;box-shadow:0 0 22px #52efff16}.google-wrap{width:100%;min-height:53px;border-radius:13px;overflow:hidden;background:#f7f9fc;border:1px solid #e4eaf2;display:flex;align-items:center;justify-content:center}.google-wrap iframe{max-width:100%!important}.google-wrap:hover{box-shadow:0 7px 24px #0004}.guest-icon{width:20px;height:20px;color:#6cf5ff}.divider{display:flex;align-items:center;gap:12px;color:#637895;font-size:10px;letter-spacing:2px;margin:20px 0}.divider:before,.divider:after{content:"";height:1px;flex:1;background:#263950}.fine{text-align:center;color:#7285a2;font-size:10px;line-height:1.8;margin-top:23px}.fine strong{color:#9eb2cf;font-weight:650}.status{display:none;margin-top:14px;padding:11px 12px;border:1px solid #5e3b5e;background:#211323;color:#ffc8db;border-radius:10px;font-size:12px;line-height:1.6;overflow-wrap:anywhere}.status.show{display:block}.scan{height:2px;width:100%;background:linear-gradient(90deg,transparent,#5ff4ff, #a56cff,transparent);margin:0 0 24px;opacity:.75}.wave{text-align:center;color:#50e9fa;letter-spacing:5px;font-size:15px;margin-top:23px;opacity:.7}@media(max-width:480px){.card{padding:29px 21px;border-radius:21px}.title{font-size:35px}.top{letter-spacing:3px}}
</style></head><body><div class="bg"></div><main class="shell"><div class="top"><span class="orb">♫</span><span>HATHOR AI / NEURAL STUDIO</span></div><div class="card"><div class="scan"></div><div class="eyebrow">CINEMATIC MUSIC GENERATION</div><h1 class="title">Your next sound<br>starts <span>here.</span></h1><p class="sub">Enter a creative space where learned musical patterns become your next composition.</p>
<div class="google-wrap" id="googleButton" aria-label="Continue with Google"></div>
<button class="btn" id="guestButton" type="button"><svg class="guest-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c.7-3.3 3.1-5 7-5s6.3 1.7 7 5"/><path d="M18.5 4.5l1 1 2-2"/></svg><span>Continue as Guest</span></button>
<div id="status" class="status" role="alert"></div><div class="divider">SECURE ACCESS</div><div class="fine"><strong>Google sign-in is verified securely.</strong><br>Guest mode is available without an account.<br>Your successful sign-ins are recorded for studio access analytics.</div><div class="wave">▂ ▄ ▆ ▃ ▇ ▄ ▂ ▅ ▇ ▃ ▆ ▂</div></div></main>
<script>
const CLIENT_ID = __GOOGLE_CLIENT_ID__;
const statusBox = document.getElementById('status');
function showStatus(message){statusBox.textContent=message;statusBox.classList.add('show');}
function busy(button, text){button.disabled=true;button.dataset.original=button.innerHTML;button.innerHTML=text;}
async function postJson(url, payload){const response=await fetch(url,{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let data={};try{data=await response.json()}catch(e){}if(!response.ok||!data.ok)throw new Error(data.error||data.detail||'Sign-in failed. Please try again.');return data;}
function handleGoogleCredential(response){postJson('/auth/google',{credential:response.credential}).then(()=>{window.location.replace('/studio')}).catch(e=>showStatus(e.message));}
function initGoogle(){if(!window.google||!google.accounts||!google.accounts.id)return false;google.accounts.id.initialize({client_id:CLIENT_ID,callback:handleGoogleCredential,auto_select:false,cancel_on_tap_outside:true});const target=document.getElementById('googleButton');if(!target.dataset.rendered){google.accounts.id.renderButton(target,{type:'standard',theme:'outline',size:'large',text:'continue_with',shape:'rectangular',logo_alignment:'left',width:360});target.dataset.rendered='yes';}return true;}
document.getElementById('guestButton').addEventListener('click',async()=>{const button=document.getElementById('guestButton');busy(button,'Creating guest session…');try{await postJson('/auth/guest',{});window.location.replace('/studio')}catch(e){button.disabled=false;button.innerHTML=button.dataset.original;showStatus(e.message)}});
// Initialize after the Google Identity script becomes available.
let tries=0;const timer=setInterval(()=>{tries++;if(initGoogle()||tries>40)clearInterval(timer)},250);
fetch('/auth/me',{credentials:'same-origin'}).then(r=>r.json()).then(d=>{if(d.authenticated)window.location.replace('/studio')}).catch(()=>{});
</script></body></html>'''

LOGIN_HTML = LOGIN_HTML.replace("__GOOGLE_CLIENT_ID__", json.dumps(GOOGLE_CLIENT_ID))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/studio", status_code=303)
    return HTMLResponse(LOGIN_HTML)

@app.get("/health")
async def health():
    return {"ok": True, "service": "Hathor AI", "model": "cinematic-model/checkpoints/best_model.keras"}

@app.get("/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    return {"authenticated": bool(user), "name": user.get("name") if user else None}

@app.post("/auth/guest")
async def guest_login(request: Request):
    user_agent = request.headers.get("user-agent", "")
    user = {"name": "Guest", "email": "", "method": "Guest", "session_id": uuid.uuid4().hex, "user_agent": user_agent}
    try:
        log_successful_login("Guest", user)
    except Exception as exc:
        print(f"Guest login tracking failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="The spreadsheet could not record this login. Please try again in a moment.")
    request.session["hathor_user"] = user
    return JSONResponse({"ok": True, "method": "Guest"})

@app.post("/auth/google")
async def google_login(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid sign-in request.")
    credential = body.get("credential") if isinstance(body, dict) else None
    if not credential or not isinstance(credential, str):
        raise HTTPException(status_code=400, detail="Google did not return a valid credential.")
    try:
        claims = id_token.verify_oauth2_token(credential, google_requests.Request(), GOOGLE_CLIENT_ID)
    except Exception:
        raise HTTPException(status_code=401, detail="Google verification failed. Please sign in again.")
    if claims.get("aud") != GOOGLE_CLIENT_ID or claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(status_code=401, detail="This Google credential is not valid for Hathor AI.")
    if claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Please use a Google account with a verified email address.")
    user = {
        "name": str(claims.get("name") or claims.get("given_name") or "Google user")[:160],
        "email": str(claims.get("email") or "")[:254],
        "method": "Google",
        "session_id": uuid.uuid4().hex,
        "user_agent": request.headers.get("user-agent", ""),
        "google_sub": str(claims.get("sub") or "")[:160],
    }
    try:
        log_successful_login("Google", user)
    except Exception as exc:
        print(f"Google login tracking failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="The spreadsheet could not record this login. Please try again in a moment.")
    request.session["hathor_user"] = user
    return JSONResponse({"ok": True, "method": "Google", "name": user["name"]})

@app.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}

@app.get("/logout")
async def logout_redirect(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)

# Protect the complete Gradio mount, including its API and websocket routes.
app = gr.mount_gradio_app(app, demo, path="/studio", auth_dependency=gradio_auth_dependency)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
