# Third-party and licensing review

This file is a practical release checklist, not legal advice.

## Luminote source license

Luminote's original source code is licensed under Apache License 2.0.

Copyright 2026 John Ryan Manning. The complete terms are in `LICENSE`, and the project attribution is in `NOTICE`.

## Deliberately excluded

- all frozen story-probe research, datasets, packets, and results;
- Hugging Face model weights and caches;
- calibration arrays and generated metadata;
- conversation and generation-integrity logs;
- virtual environments, caches, package metadata, and machine-specific files;
- the private reaction meme/GIF/image collection and source PSD;
- copied/labeled affective-circumplex artwork with undocumented redistribution rights;
- screenshots or videos that may expose private conversations or unlicensed imagery.

## Separately licensed dependencies

Python and JavaScript dependencies retain their own licenses and are installed from their normal package sources; none are vendored here. Review the dependency versions in `pyproject.toml` when producing a formal notice.

The default `Qwen/Qwen2.5-1.5B-Instruct` model is not bundled. Its official Hugging Face repository identifies it as Apache-2.0. Users download it separately and remain responsible for its license and acceptable-use requirements:

- <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE>

## Included-source audit

The public package contains Luminote's original Python, JavaScript, HTML, CSS, configuration, tests, and documentation. No third-party source code, copied examples, vendored libraries, model implementation, or generated dependency bundle was identified in the included files. Consequently, no included source file requires a separate third-party copyright header or NOTICE entry.

The standard Apache License text itself is included as `LICENSE`. Package dependencies are referenced by name and version range only and are installed separately; their license texts are not copied into this repository.

## Publication checklist

- [x] Add Apache License 2.0 for Luminote's original source code.
- [x] Add `NOTICE` with Copyright 2026 John Ryan Manning.
- [ ] Confirm the repository visibility and owner account.
- [ ] Run the secret and machine-path scan described in the release handoff.
- [ ] Confirm that `research/`, logs, model weights, prototypes, and private reaction assets are absent.
- [ ] Review every screenshot/video for private text and third-party imagery.
- [ ] Confirm that the README accurately distinguishes the public color-only build from the private reaction-asset demo.
