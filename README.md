# Luminote / RiverHUD

Luminote is a local experimental chat interface where the page background acts as **machine body language**. RiverHUD samples a local language model's hidden state at the assistant-response boundary, compares it with prompted concept prototypes, and maps the result into an exploratory valence/arousal “weather” display.

The text is the model's generated response. The color is an activation-derived visualization. Previous assistant turns retain their exact saved weather and diagnostics and can be revisited without rerunning the model.

This public hackathon package contains the runnable color-weather prototype. The private reaction GIF/meme collection and the separate story-probe research are intentionally excluded.

## Technology

- Python 3.11+
- FastAPI and Uvicorn
- PyTorch with Apple MPS, CUDA, or CPU fallback
- Hugging Face Transformers
- `Qwen/Qwen2.5-1.5B-Instruct`
- NumPy and Pillow
- Vanilla HTML, CSS, and JavaScript

All chat inference and application logging run locally. Internet access is needed initially to install Python packages and download the model from Hugging Face.

## Install

macOS or Linux:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/luminote.git
cd luminote
./scripts/setup.sh
./scripts/calibrate.sh
```

`setup.sh` creates `.venv`, installs the package and test dependencies, and generates an original public-safe color wheel. `calibrate.sh` downloads the configured model on first use and creates local prototype files. Calibration can take several minutes.

## Launch

```bash
./scripts/run.sh
```

Open <http://127.0.0.1:8765>. Stop the local server with `Control-C` in the terminal that launched it.

Optional model and layer settings:

```bash
LUMINOTE_MODEL=Qwen/Qwen2.5-1.5B-Instruct LUMINOTE_LAYER=24 ./scripts/run.sh
```

If port 8765 is already in use:

```bash
LUMINOTE_PORT=8877 ./scripts/run.sh
```

If the model or concept configuration changes, rerun `./scripts/calibrate.sh`.

## Hardware expectations

The prototype was developed on an Apple M4 Pro with 24 GB unified memory. Apple Silicon with at least 16 GB memory is recommended. CUDA is supported when PyTorch detects it. CPU fallback works but generation and calibration may be slow.

Allow roughly 5 GB of free disk space for the Python environment, model cache, and generated calibration files. The initial Qwen model download is approximately 3.1 GB.

## Tests

The included tests do not download or load the language model:

```bash
.venv/bin/pytest
node --test tests/*.test.mjs
```

## Optional reaction assets

The interface supports deterministic nearest-neighbor reaction images, but the hackathon creator's meme/GIF/PSD collection is not distributed because its third-party redistribution rights are unclear. The included empty manifest makes the app fall back cleanly to color weather.

To add reactions, use only images you created or are licensed to redistribute and follow [ASSETS.md](ASSETS.md). Never commit the private `originals/` or `maps/` directories by accident.

## Screenshots and demo media

The `screenshots/` directory is intentionally empty. Before adding media:

1. use only screenshots and video you created;
2. remove private conversations, usernames, paths, notifications, and account information;
3. confirm that every visible reaction image is licensed for public redistribution; and
4. prefer showing the public-safe color-only build if asset rights are uncertain.

## Local data and privacy

Conversation and generation-integrity logs are written under `logs/` and ignored by Git. They can contain complete prompts and model responses. Calibration arrays are written under `data/prototypes/` and ignored by Git. Model weights stay in the Hugging Face cache and are never part of this repository.

The app has no accounts, cloud chat service, telemetry, or remote analytics. Hugging Face is contacted only when model files are not already cached.

## How the estimate works

1. Render the conversation with the model tokenizer's native chat template.
2. Run a hidden-state forward pass at the assistant-response boundary.
3. Compare that activation with locally calibrated, mean-centered concept prototypes using cosine similarity.
4. Combine the strongest positive matches into an exploratory valence/arousal coordinate.
5. Map that coordinate to a color and preserve the complete snapshot with its assistant turn.

The editable prototype concepts and researcher-assigned coordinates live in `data/emotion_coordinates.json`. Calibration outputs are model-specific and are generated locally rather than distributed.

## Scientific limitations

- This is an exploratory interpretability interface, not a validated affect detector.
- Prompted concepts may capture wording, topic, prompt format, or other correlated features rather than affect.
- Valence/arousal coordinates and confidence calculations are engineering choices, not ground truth or calibrated probabilities.
- Results may change with prompts, model versions, templates, sampling settings, layer choice, or hardware numerics.
- A single boundary activation compresses a high-dimensional computation into two numbers and discards nearly all information.
- The output does not demonstrate emotion, intention, deception, sentience, consciousness, or a model's subjective experience.
- Do not use it for clinical, psychological, employment, educational, legal, safety-critical, or high-stakes judgments.

Describe the output as “activation-derived affective weather,” an “affect estimate,” or a “valence/arousal visualization.”

## Repository layout

```text
luminote/             local model, affect estimator, color mapper, API, logging
frontend/             browser chat interface and per-turn diagnostics
data/                 public concept configuration; generated files are ignored
luminote-reactions/   empty public manifest and optional user-supplied assets
scripts/              setup, calibration, local launch, and optional asset tooling
tests/                model-free unit tests
screenshots/          intentionally empty public-media directory
```

## License

Luminote's original source code is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 John Ryan Manning. See [NOTICE](NOTICE) and [THIRD_PARTY_AND_LICENSING.md](THIRD_PARTY_AND_LICENSING.md) for attribution, exclusions, and separately licensed dependencies.

The locally curated reaction images, GIFs, memes, and PSD source map shown in the private demo are **not redistributed**. The public build contains an empty reaction manifest and defaults to color-weather mode. Users may add only assets they created or are licensed to redistribute.

The default model is downloaded separately and is listed by its official model card as Apache-2.0: [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [model license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE).
