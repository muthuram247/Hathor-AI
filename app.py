import os
import json
import time
import random
import wave
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pretty_midi
import tensorflow as tf


# ==========================================
# HATHOR AI — MODEL AND FILE CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "hathor_best.keras"
DATA_PATH = BASE / "data" / "processed" / "hathor_notes.npz"
VOCAB_PATH = BASE / "data" / "processed" / "vocabulary.json"
DURATION_PATH = BASE / "data" / "processed" / "duration_bins.json"

OUTPUT_DIR = BASE / "outputs" / "midi"
AUDIO_DIR = BASE / "outputs" / "audio"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


with open(VOCAB_PATH, encoding="utf-8") as f:
    vocabulary = json.load(f)

with open(DURATION_PATH, encoding="utf-8") as f:
    DURATION_BINS = json.load(f)

with np.load(DATA_PATH, allow_pickle=False) as data:
    X = data["X"]
    sequence_length = int(data["sequence_length"][0])


# Load the trained model once when the service starts.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# ==========================================
# MUSIC GENERATION
# ==========================================

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


# ==========================================
# MIDI EXPORT WITH SELECTABLE INSTRUMENT
# ==========================================

def create_midi(notes, tempo, instrument_name):
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))

    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)

    current_time = 0.0

    for pitch, duration_idx in notes:
        duration_idx = int(duration_idx)

        if not 0 <= duration_idx < len(DURATION_BINS):
            raise ValueError("Generated duration index is invalid.")

        duration = max(
            0.08,
            min(float(DURATION_BINS[duration_idx]), 2.5)
        )

        note = pretty_midi.Note(
            velocity=90,
            pitch=int(np.clip(pitch, 0, 127)),
            start=current_time,
            end=current_time + duration
        )

        instrument.notes.append(note)
        current_time += duration

    midi.instruments.append(instrument)

    stamp = int(time.time())
    unique_id = random.randrange(100000)

    output_path = OUTPUT_DIR / (
        f"hathor_{stamp}_{unique_id}.mid"
    )

    midi.write(str(output_path))

    # Validate the exported MIDI.
    check = pretty_midi.PrettyMIDI(str(output_path))

    if not check.instruments or not check.instruments[0].notes:
        raise RuntimeError(
            "The generated MIDI file could not be validated."
        )

    return midi, str(output_path)


# ==========================================
# SIMPLE BUILT-IN AUDIO PREVIEW
# ==========================================

def instrument_waveform(phase, instrument_name):
    """
    Create simple, distinct preview timbres without
    requiring an external synthesizer installation.
    """

    name = instrument_name.lower()

    if "flute" in name:
        return (
            np.sin(phase)
            + 0.12 * np.sin(2 * phase)
        ) / 1.12

    if "violin" in name or "cello" in name or "strings" in name:
        return (
            np.sin(phase)
            + 0.45 * np.sin(2 * phase)
            + 0.22 * np.sin(3 * phase)
            + 0.10 * np.sin(4 * phase)
        ) / 1.77

    if "guitar" in name:
        return (
            np.sin(phase)
            + 0.35 * np.sin(2 * phase)
            + 0.18 * np.sin(3 * phase)
        ) / 1.53

    if "trumpet" in name or "sax" in name:
        return (
            np.sin(phase)
            + 0.55 * np.sin(2 * phase)
            + 0.35 * np.sin(3 * phase)
            + 0.20 * np.sin(4 * phase)
        ) / 2.10

    if "organ" in name:
        return (
            np.sin(phase)
            + 0.50 * np.sin(2 * phase)
            + 0.25 * np.sin(3 * phase)
        ) / 1.75

    # Piano-like default tone.
    return (
        np.sin(phase)
        + 0.30 * np.sin(2 * phase)
        + 0.12 * np.sin(3 * phase)
    ) / 1.42


def make_audio_preview(notes, tempo, instrument_name, audio_path):
    """
    Produce a lightweight WAV preview with a basic
    synthesized tone. This is not a sampled instrument library.
    """

    sample_rate = 22050
    seconds_per_beat = 60.0 / float(tempo)

    # The training duration bins represent note durations
    # in beats for this preview.
    durations = []

    for _, duration_idx in notes:
        duration_idx = int(duration_idx)

        if not 0 <= duration_idx < len(DURATION_BINS):
            raise ValueError("Generated duration index is invalid.")

        beats = max(
            0.08,
            min(float(DURATION_BINS[duration_idx]), 2.5)
        )

        durations.append(beats * seconds_per_beat)

    total_seconds = sum(durations) + 0.2

    # Prevent accidental creation of excessively large WAV files.
    total_seconds = min(total_seconds, 180.0)
    total_samples = int(total_seconds * sample_rate)

    audio = np.zeros(total_samples, dtype=np.float32)
    current_time = 0.0

    for (pitch, _), duration in zip(notes, durations):
        if current_time >= total_seconds:
            break

        duration = min(duration, total_seconds - current_time)
        note_samples = int(duration * sample_rate)

        if note_samples < 2:
            current_time += duration
            continue

        start = int(current_time * sample_rate)
        end = min(start + note_samples, total_samples)
        note_samples = end - start

        if note_samples < 2:
            break

        frequency = 440.0 * (
            2.0 ** ((int(pitch) - 69) / 12.0)
        )

        time_axis = np.arange(note_samples) / sample_rate
        phase = 2.0 * np.pi * frequency * time_axis

        tone = instrument_waveform(phase, instrument_name)

        # Simple attack and release envelope.
        attack_samples = min(
            int(0.015 * sample_rate),
            note_samples // 3
        )
        release_samples = min(
            int(0.06 * sample_rate),
            note_samples // 3
        )

        envelope = np.ones(note_samples, dtype=np.float32)

        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(
                0.0, 1.0, attack_samples, endpoint=False
            )

        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(
                1.0, 0.0, release_samples
            )

        audio[start:end] += (
            tone[:note_samples] * envelope
        ).astype(np.float32)

        current_time += duration

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

    if peak > 0:
        audio = audio / peak * 0.85

    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return str(audio_path)


# ==========================================
# GENERATE MUSIC + UPDATE SESSION HISTORY
# ==========================================

def clean_notes(notes, tempo):
    """Improve pitch range and timing before playback and export."""

    beat = 60.0 / float(tempo)
    grid = beat / 2.0  # Eighth-note grid
    cleaned = []

    current_time = 0.0

    for pitch, duration_idx in notes:
        pitch = int(pitch)
        duration_idx = int(duration_idx)

        if not 0 <= duration_idx < len(DURATION_BINS):
            continue

        # Raise the melody one octave, avoiding extreme high notes.
        pitch = int(np.clip(pitch + 12, 48, 96))

        # Treat learned duration bins as beats.
        duration_beats = float(DURATION_BINS[duration_idx])
        duration_beats = float(np.clip(duration_beats, 0.25, 1.0))

        # Snap note starts to a regular rhythmic grid.
        current_time = round(current_time / grid) * grid

        cleaned.append((pitch, duration_idx))
        current_time += duration_beats * beat

    return cleaned

def generate_music(
    length,
    temperature,
    tempo,
    instrument_name,
    history_items
):
    try:
        notes = generate_tokens(int(length), float(temperature))

        midi, midi_path = create_midi(
            notes,
            float(tempo),
            instrument_name
        )

        audio_name = Path(midi_path).with_suffix(".wav").name
        audio_path = AUDIO_DIR / audio_name

        make_audio_preview(
            notes,
            float(tempo),
            instrument_name,
            audio_path
        )

        label = (
            f"{datetime.now().strftime('%H:%M:%S')} · "
            f"{instrument_name} · {len(notes)} notes"
        )

        history_items = list(history_items or [])

        history_items.insert(0, {
            "label": label,
            "midi": midi_path,
            "audio": str(audio_path),
            "status": (
                f"Generated {len(notes)} notes using "
                f"{instrument_name} at {int(tempo)} BPM."
            )
        })

        # Keep the most recent 10 generations for this session.
        history_items = history_items[:10]

        history_choices = [
            item["label"] for item in history_items
        ]

        return (
            midi_path,
            str(audio_path),
            (
                f"Composition ready.\n"
                f"Instrument: {instrument_name}\n"
                f"Notes: {len(notes)}\n"
                f"Tempo: {int(tempo)} BPM"
            ),
            history_items,
            gr.update(
                choices=history_choices,
                value=label
            )
        )

    except Exception as exc:
        raise gr.Error(f"Music generation failed: {exc}")


def load_history(label, history_items):
    for item in history_items or []:
        if item["label"] == label:
            return (
                item["midi"],
                item["audio"],
                item["status"]
            )

    return (
        None,
        None,
        "Choose a composition from your generation history."
    )


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

#result-file, #audio-preview {
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


# ==========================================
# USER INTERFACE
# ==========================================

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
            Create original sequences with your AI music model,
            choose a playback instrument, and export your work as MIDI.
        </p>
        <div id="hathor-wave" aria-hidden="true">
            ▁ ▃ ▅ ▂ ▇ ▄ ▆ ▂ ▅ ▇ ▃ ▆ ▂ ▄ ▇ ▃ ▁
        </div>
    </header>
    """)

    history = gr.State([])

    with gr.Row(equal_height=False):

        with gr.Column(scale=6):
            with gr.Group(elem_classes=["hathor-panel"]):

                gr.HTML("""
                <div class="hathor-heading">
                    ✦ Create your melody
                </div>
                <div class="hathor-description">
                    Shape your composition, select a playback instrument,
                    and generate a new musical idea.
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
                    info=(
                        "Lower values are more predictable; "
                        "higher values are more varied"
                    )
                )

                tempo = gr.Slider(
                    minimum=60,
                    maximum=180,
                    value=110,
                    step=5,
                    label="Tempo (BPM)",
                    info="Playback speed for your composition"
                )

                instrument = gr.Dropdown(
                    choices=[
                        "Acoustic Grand Piano",
                        "Violin",
                        "Cello",
                        "Flute",
                        "Acoustic Guitar (nylon)",
                        "Electric Guitar (clean)",
                        "Trumpet",
                        "Soprano Sax",
                        "Church Organ",
                        "Synth Strings 1"
                    ],
                    value="Acoustic Grand Piano",
                    label="Playback instrument",
                    info="Choose the sound for your melody."
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
                    Listen to your preview directly here or download
                    the MIDI file for use in a compatible music app.
                </div>
                """)

                audio_result = gr.Audio(
                    label="Listen to your composition",
                    type="filepath",
                    interactive=False,
                    elem_id="audio-preview"
                )

                result = gr.File(
                    label="Download MIDI composition",
                    elem_id="result-file",
                    interactive=False
                )

                status = gr.Textbox(
                    label="Studio status",
                    value="Ready to create your first melody.",
                    lines=4,
                    interactive=False,
                    elem_id="status-box"
                )

                gr.Markdown(
                    "**EXPORT** · MIDI (.mid) + WAV (.wav)  \n"
                    "**HISTORY** · Latest 10 compositions in this session"
                )

    with gr.Group(elem_classes=["hathor-panel"]):
        gr.HTML("""
        <div class="hathor-heading">
            ◷ Generation history
        </div>
        <div class="hathor-description">
            Reopen a composition created during this session.
            Select a previous entry to restore its audio and MIDI file.
        </div>
        """)

        history_picker = gr.Dropdown(
            choices=[],
            label="Previous compositions",
            interactive=True,
            info="Choose a composition from this session."
        )

    gr.HTML("""
    <div class="hathor-footer">
        HATHOR AI · FROM NEURAL PATTERNS TO MUSICAL IDEAS<br>
        An experimental AI music-generation project
    </div>
    """)

    generate.click(
        fn=generate_music,
        inputs=[
            length,
            temperature,
            tempo,
            instrument,
            history
        ],
        outputs=[
            result,
            audio_result,
            status,
            history,
            history_picker
        ]
    )

    history_picker.change(
        fn=load_history,
        inputs=[
            history_picker,
            history
        ],
        outputs=[
            result,
            audio_result,
            status
        ]
    )


# ==========================================
# START WEB SERVER
# ==========================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000"))
        )
                
