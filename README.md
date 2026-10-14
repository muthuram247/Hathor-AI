# Hathor AI — AI Music Generator

**Unleash Your Own Music.**

## Objective
Build a small LSTM model that learns next-note patterns from classical MIDI files and exports a newly generated note sequence as MIDI.

## Workflow
1. Read classical scores from the installed `music21` corpus and export usable scores to MIDI.
2. Extract pitches and quantized durations from valid MIDI files.
3. Build fixed-length sequences and next-token targets.
4. Train a TensorFlow/Keras LSTM with a validation split.
5. Sample new notes using temperature sampling.
6. Export and validate a MIDI file; WAV rendering is optional.

## Dataset and licensing
This notebook uses Bach entries available in the installed `music21` corpus and converts readable scores to MIDI. Bach's compositions are public domain in many jurisdictions, but individual digital editions and corpus files can have separate terms. Check the specific source file notices before redistributing corpus files. Cite the music21 project and identify the actual files used in your report. Do not claim every file has the same licence.

## Actual run metrics
- Valid MIDI files processed: 60
- Extracted note events: 17254
- Training sequences: 15334
- Vocabulary size: 323
- Epochs completed: 12
- Final training loss: 3.0292530059814453
- Final validation loss: 3.3841021060943604

These values are populated only after the notebook has actually run.

## Requirements
Python, TensorFlow/Keras, NumPy, music21, pretty_midi, matplotlib, and optionally Gradio.

## Limitations
This is a small educational next-note model. Output may sound repetitive or musically imperfect. A low validation loss does not by itself prove musical quality. WAV rendering requires a compatible synthesizer and soundfont; MIDI export is the core supported output.

## Reproduce on Google Colab
Open `notebooks/Hathor_AI_Training.ipynb`, run cells from top to bottom, and inspect the metrics and generated MIDI. GPU access is optional and not guaranteed.

## Screenshots for the academic report
Capture: (1) dataset counts, (2) model summary, (3) training/validation loss plot, (4) successful MIDI export, and (5) the Gradio interface if deployed.
