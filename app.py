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

with gr.Blocks(title="Hathor AI") as demo:
    gr.Markdown("# Hathor AI")
    gr.Markdown("### Unleash Your Own Music")
    gr.Markdown(
        "Generate a new piano melody using an LSTM trained on classical MIDI music. "
        "Adjust the settings, generate a melody, and download the MIDI file."
    )

    with gr.Row():
        length = gr.Slider(
            minimum=32, maximum=240, value=120, step=8,
            label="Generation length (notes)"
        )
        temperature = gr.Slider(
            minimum=0.3, maximum=1.5, value=0.9, step=0.1,
            label="Creativity (temperature)"
        )

    tempo = gr.Slider(
        minimum=60, maximum=180, value=110, step=5,
        label="Tempo (BPM)"
    )
    generate = gr.Button("Generate Music", variant="primary")
    result = gr.File(label="Download your MIDI melody")
    status = gr.Textbox(label="Status", interactive=False)

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
