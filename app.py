import os
import json
import time
import random
from pathlib import Path

import gradio as gr
import numpy as np
import pretty_midi
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "hathor_best.keras"
DATA_PATH = BASE / "data" / "processed" / "hathor_notes.npz"
VOCAB_PATH = BASE / "data" / "processed" / "vocabulary.json"
DURATION_PATH = BASE / "data" / "processed" / "duration_bins.json"
OUTPUT_DIR = BASE / "outputs" / "midi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(VOCAB_PATH, encoding="utf-8") as f:
    vocabulary = json.load(f)

with open(DURATION_PATH, encoding="utf-8") as f:
    DURATION_BINS = json.load(f)

with np.load(DATA_PATH, allow_pickle=False) as data:
    X = data["X"]
    sequence_length = int(data["sequence_length"][0])

# The model is loaded once when the web service starts.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def sample_next_token(probabilities, temperature):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    temperature = max(0.1, float(temperature))
    logits = np.log(np.maximum(probabilities, 1e-9)) / temperature
    logits -= np.max(logits)
    scaled = np.exp(logits)
    scaled /= scaled.sum()
    return int(np.random.choice(len(scaled), p=scaled))

def generate_tokens(length, temperature):
    seed_index = random.randrange(len(X))
    window = list(X[seed_index][-sequence_length:])
    generated = []

    for _ in range(int(length)):
        inp = np.asarray([window], dtype=np.int32)
        probabilities = model.predict(inp, verbose=0)[0]
        next_id = sample_next_token(probabilities, temperature)
        generated.append(next_id)
        window = window[1:] + [next_id]

    return [vocabulary[token_id] for token_id in generated]

def create_midi(notes, tempo):
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )
    current_time = 0.0

    for pitch, duration_idx in notes:
        duration_idx = int(duration_idx)
        if not 0 <= duration_idx < len(DURATION_BINS):
            raise ValueError("Generated duration index is invalid.")

        duration = max(0.08, min(float(DURATION_BINS[duration_idx]), 2.5))
        note = pretty_midi.Note(
            velocity=90,
            pitch=int(np.clip(pitch, 0, 127)),
            start=current_time,
            end=current_time + duration
        )
        instrument.notes.append(note)
        current_time += duration

    midi.instruments.append(instrument)
    output_path = OUTPUT_DIR / f"hathor_{int(time.time())}_{random.randrange(10000)}.mid"
    midi.write(str(output_path))

    # Check that the exported MIDI can be read.
    check = pretty_midi.PrettyMIDI(str(output_path))
    if not check.instruments or not check.instruments[0].notes:
        raise RuntimeError("The generated MIDI file could not be validated.")

    return str(output_path)

def generate_music(length, temperature, tempo):
    try:
        notes = generate_tokens(int(length), float(temperature))
        path = create_midi(notes, float(tempo))
        return path, f"Successfully generated {len(notes)} notes."
    except Exception as exc:
        raise gr.Error(f"Music generation failed: {exc}")
with gr.Blocks(
    title="Hathor AI | Music Studio",
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css="""
    .gradio-container {
        max-width: 1050px !important;
        margin: auto !important;
        background: #080d18 !important;
    }

    #hero {
        text-align: center;
        padding: 46px 20px 32px;
        border: 1px solid #25354b;
        border-radius: 22px;
        background: linear-gradient(135deg, #101d32, #0a1020);
        margin-bottom: 22px;
    }

    #hero h1 {
        font-size: clamp(36px, 7vw, 60px);
        font-weight: 800;
        letter-spacing: -2px;
        margin: 12px 0;
        color: #f1f7ff;
    }

    #hero p {
        color: #a9bbd2;
        font-size: 16px;
        line-height: 1.7;
    }

    .eyebrow {
        color: #54e5ef;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 3px;
    }

    .panel {
        border: 1px solid #25354b !important;
        border-radius: 18px !important;
        padding: 22px !important;
        background: #101827 !important;
    }

    #generate-btn {
        border-radius: 12px !important;
        min-height: 54px !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }

    #footer {
        text-align: center;
        color: #8091a8;
        font-size: 12px;
        padding: 22px 8px;
    }

    footer {
        display: none !important;
    }
    """
) as demo:

    gr.HTML("""
    <div id="hero">
        <div class="eyebrow">HATHOR AI · MUSIC INTELLIGENCE</div>
        <h1>Make room for <span style="color:#54e5ef">music.</span></h1>
        <p>
            An experimental AI music studio powered by an LSTM neural network.
            Create original piano sequences, shape their feel, and export MIDI.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("## 🎹 Create your melody")
                gr.Markdown(
                    "Adjust the controls below, then generate a new composition."
                )

                length = gr.Slider(
                    minimum=32,
                    maximum=240,
                    value=120,
                    step=8,
                    label="Melody length",
                    info="More notes create a longer sequence."
                )

                temperature = gr.Slider(
                    minimum=0.3,
                    maximum=1.5,
                    value=0.9,
                    step=0.1,
                    label="Creativity",
                    info="Lower values are more predictable; "
                         "higher values are more varied."
                )

                tempo = gr.Slider(
                    minimum=60,
                    maximum=180,
                    value=110,
                    step=5,
                    label="Tempo (BPM)",
                    info="Choose how fast the melody plays."
                )

                generate = gr.Button(
                    "✦  Generate Music",
                    variant="primary",
                    elem_id="generate-btn"
                )

        with gr.Column(scale=2):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("## Your composition")
                gr.Markdown(
                    "Your generated MIDI file will appear here. "
                    "Download it and open it in a MIDI-compatible "
                    "music player or digital audio workstation."
                )

                result = gr.File(
                    label="Generated MIDI file",
                    interactive=False
                )

                status = gr.Textbox(
                    label="Generation status",
                    value="Ready to create.",
                    interactive=False,
                    lines=2
                )

                gr.Markdown(
                    "**File format:** MIDI (.mid)  \n"
                    "**Instrument:** Acoustic Grand Piano"
                )

    gr.HTML("""
    <div id="footer">
        HATHOR AI · From neural patterns to musical ideas<br>
        An experimental AI music-generation project
    </div>
    """)

    generate.click(
        fn=generate_music,
        inputs=[length, temperature, tempo],
        outputs=[result, status]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000"))
                )
