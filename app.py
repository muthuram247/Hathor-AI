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

        # ==========================================
# HATHOR AI — MIDNIGHT STUDIO INTERFACE
# ==========================================

MIDNIGHT_CSS = """
.gradio-container {
    max-width: 1120px !important;
    margin: auto !important;
    background: #080d18 !important;
    color: #f2f7ff !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

body, .main {
    background: #080d18 !important;
}

footer {
    display: none !important;
}

#hathor-header {
    background:
        radial-gradient(ellipse at top, #18314c 0%, transparent 70%),
        #0b1220;
    border: 1px solid #263950;
    border-radius: 24px;
    padding: 30px 24px;
    margin: 8px 0 22px;
    text-align: center;
}

#hathor-brand {
    color: #55e5ef;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 4px;
    margin-bottom: 16px;
}

#hathor-header h1 {
    color: #f2f7ff;
    font-size: clamp(34px, 6vw, 58px);
    line-height: 1.1;
    letter-spacing: -2px;
    margin: 0 0 14px;
}

#hathor-header h1 span {
    color: #55e5ef;
}

#hathor-header p {
    max-width: 590px;
    margin: auto;
    color: #a5b7d0;
    font-size: 15px;
    line-height: 1.8;
}

#hathor-wave {
    color: #55e5ef;
    letter-spacing: 5px;
    font-size: 23px;
    margin-top: 20px;
    overflow: hidden;
    white-space: nowrap;
}

.hathor-panel {
    background: #101a2b !important;
    border: 1px solid #263950 !important;
    border-radius: 20px !important;
    padding: 22px !important;
}

.hathor-heading {
    color: #f2f7ff;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 8px;
}

.hathor-description {
    color: #9aacc4;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 20px;
}

.hathor-footer {
    text-align: center;
    color: #8294ac;
    font-size: 12px;
    line-height: 1.9;
    padding: 24px 8px 12px;
}

#generate-btn {
    background: #55e5ef !important;
    color: #07111d !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 54px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    transition: transform .15s ease;
}

#generate-btn:hover {
    filter: brightness(1.08);
}

button, input, textarea {
    border-radius: 10px !important;
}

label, .label-wrap span {
    color: #dce8f8 !important;
}

input[type="range"] {
    accent-color: #55e5ef !important;
}

#result-file {
    border: 1px dashed #38516d !important;
    border-radius: 14px !important;
    background: #0b1423 !important;
}

#status-box textarea {
    color: #b6c8df !important;
    background: #0b1423 !important;
    border: 1px solid #263950 !important;
}

@media (max-width: 640px) {
    .gradio-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }

    #hathor-header {
        padding: 26px 16px;
        border-radius: 18px;
    }

    #hathor-header h1 {
        font-size: 37px;
        letter-spacing: -1.4px;
    }

    #hathor-header p {
        font-size: 14px;
    }

    .hathor-panel {
        padding: 17px !important;
    }
}
"""

with gr.Blocks(
    title="Hathor AI | Midnight Studio",
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=MIDNIGHT_CSS
) as demo:

    gr.HTML("""
    <header id="hathor-header">
        <div id="hathor-brand">♫ HATHOR AI · MUSIC STUDIO</div>
        <h1>Make room for <span>music.</span></h1>
        <p>
            Turn neural patterns into musical ideas.
            Create original piano sequences with your AI music model
            and export your compositions as MIDI.
        </p>
        <div id="hathor-wave" aria-hidden="true">
            ▁ ▃ ▅ ▂ ▇ ▄ ▆ ▂ ▅ ▇ ▃ ▆ ▂ ▄ ▇ ▃ ▁
        </div>
    </header>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=6):
            with gr.Group(elem_classes=["hathor-panel"]):
                gr.HTML("""
                <div class="hathor-heading">
                    ✦ Create your melody
                </div>
                <div class="hathor-description">
                    Shape the length and creativity of your next composition.
                    Adjust the settings and generate a new MIDI file.
                </div>
                """)

                length = gr.Slider(
                    minimum=32,
                    maximum=240,
                    value=120,
                    step=8,
                    label="Melody length",
                    info="Number of generated note events"
                )

                temperature = gr.Slider(
                    minimum=0.3,
                    maximum=1.5,
                    value=0.9,
                    step=0.1,
                    label="Creativity",
                    info="Lower values are more predictable; "
                         "higher values are more varied"
                )

                tempo = gr.Slider(
                    minimum=60,
                    maximum=180,
                    value=110,
                    step=5,
                    label="Tempo (BPM)",
                    info="Playback speed for your composition"
                )

                generate = gr.Button(
                    "✦  Generate Music",
                    variant="primary",
                    elem_id="generate-btn"
                )

        with gr.Column(scale=5):
            with gr.Group(elem_classes=["hathor-panel"]):
                gr.HTML("""
                <div class="hathor-heading">
                    Your composition
                </div>
                <div class="hathor-description">
                    When generation finishes, download your MIDI file
                    and open it in a compatible music player or DAW.
                </div>
                """)

                result = gr.File(
                    label="Generated MIDI file",
                    elem_id="result-file",
                    interactive=False
                )

                status = gr.Textbox(
                    label="Studio status",
                    value="Ready to create your first melody.",
                    lines=3,
                    interactive=False,
                    elem_id="status-box"
                )

                gr.Markdown(
                    "**FORMAT** · MIDI (.mid)  \n"
                    "**INSTRUMENT** · Acoustic Grand Piano"
                )

    gr.HTML("""
    <div class="hathor-footer">
        HATHOR AI · FROM NEURAL PATTERNS TO MUSICAL IDEAS<br>
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
