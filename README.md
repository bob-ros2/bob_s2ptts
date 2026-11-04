# bob_s2ptts

A ROS 2 wrapper for the **Fish Audio S2-Pro** model (fishaudio/s2-pro): streaming text
aggregation, voice cloning (Eva), raw audio bridging and a Qt-based GUI for real-time
parameter tuning. Topics and node interface follow the standard launch graph
(filter → valve → tts).

> **License note**: the S2-Pro weights are under the **Fish Audio Research License**
> (research/non-commercial only, commercial use needs a separate deal with Fish Audio).
> The ROS wrapper code here stays Apache-2.0. Gated download: accept the terms on
> https://huggingface.co/fishaudio/s2-pro with your HF account first.

## Quick Start

This package is self-contained; only the engine clone and the gated weights come
from elsewhere.

```bash
# 1. Engine (upstream clone pinned to the same commit as the Dockerfile's
#    FISH_SPEECH_REF) + CUDA wheels, in a venv of your choice
git clone https://github.com/fishaudio/fish-speech.git
git -C fish-speech checkout 0f8a12e8a651062fdee9c33a894bdccd3a8513f1
python3 -m venv venv
venv/bin/pip install -e fish-speech[cu128] --extra-index-url https://download.pytorch.org/whl/cu128
venv/bin/pip install -r ros2_ws/src/bob_s2ptts/requirements.txt

# 2. Weights (gated, needs HF_TOKEN with accepted Research License);
#    lands in the package's models/ dir, which compose mounts as /models
HF_HOME=ros2_ws/src/bob_s2ptts/models hf download fishaudio/s2-pro

# 3. Build the workspace
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select bob_s2ptts

# 4. Run
source install/setup.bash
python3 -m bob_s2ptts.tts_node   # or:
ros2 run bob_s2ptts tts --ros-args -p model_dir:=ros2_ws/src/bob_s2ptts/models \
    --ros-args -r __ns:=/bob
ros2 topic pub --once /bob/text std_msgs/msg/String "{data: 'Hallo, ich bin Eva.'}"
```

The launch anchor in `/blue/dev/s2pro/s2pro.yaml` wires the standard graph (filter → valve → tts).

### Build mirrors (optional)

`docker-compose.yaml` build args can point the image build at a **Sonatype Nexus**
on the LAN: `PIP_INDEX_URL`/`PIP_TRUSTED_HOST` (PyPI group) and
`APT_UBUNTU_MIRROR`/`APT_SECURITY_MIRROR` (apt proxies). No Nexus? **Comment the
args out** — the Dockerfile falls back to stock Ubuntu/PyPI sources (the ARGs
default to empty). Heads-up: our `debian-proxy` currently mirrors Debian only, so
the apt args stay commented until an Ubuntu (jammy) remote exists in Nexus.

## Topics

| Topic | Type | Direction | Description |
| :--- | :--- | :--- | :--- |
| `text` | `std_msgs/String` | sub | Streaming text chunks, aggregated to sentences |
| `spoken_text` | `std_msgs/String` | pub | The sentence being spoken |
| `audio_raw` | `Int16MultiArray` | pub | PCM int16 mono at 44100 Hz (only processed if a subscriber is connected) |
| `speaking_flag` | `std_msgs/Bool` | pub | Latched (TRANSIENT_LOCAL) speaking state |

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | `string` | `fishaudio/s2-pro` | HF repo id or absolute path to a local checkpoint dir. Env: `S2PTTS_MODEL_ID` |
| `model_dir` | `string` | `./models` | HF cache/checkpoint directory. Env: `S2PTTS_MODEL_DIR` |
| `sentence_delimiters` | `string_array` | `[". ", "! ", "? "]` | Split points; `[""]` disables (flush_timeout only) |
| `flush_timeout` | `int` | `700` | ms of text inactivity before buffer flush |
| `substitute` | `string_array` | `[""]` | `[regex, replacement]` pairs. **Must not strip `[` / `]`** — they carry the model's expression tags |
| `temperature` | `double` | `0.7` | Schema bounds 0.1..1.0, clamped |
| `top_p` | `double` | `0.7` | Schema bounds 0.1..1.0, clamped |
| `top_k` | `int` | `30` | 0 disables the cap |
| `chunk_length` | `int` | `200` | Iterative prompt chunk, bounds 100..1000 |
| `max_new_tokens` | `int` | `1024` | Audio frames per chunk; 0 = model context cap |
| `seed` | `int` | `-1` | `-1` = random each call |
| `normalize` | `bool` | `true` | Text normalization for numbers |
| `precision` | `string` | `bfloat16` | `bfloat16` or `float16` (pre-SM80 GPUs) |
| `compile` | `bool` | `false` | torch.compile, restart required |
| `warmup` | `bool` | `true` | Dummy synthesis at startup |
| `control_instructions` | `string` | `""` in code, `[excited]` in the shipped configs | Fish inline tags (e.g. `[whisper]`), **prepended** to every synthesized sentence; `""` = off |
| `voice_ref_audio` | `string` | package `config/eva_44khz_de.wav` | Clone reference; empty = random timbre |
| `voice_ref_text` | `string` | package `config/voice_ref_text_de.txt` | Transcript or transcript file path |
| `play` | `bool` | `true` | Immediate playback |
| `player` | `string` | `sys` | `sys` (sounddevice) or executable (e.g. `aplay`) |
| `audio_device` | `string` | `""` | sounddevice device ID/name |
| `target_sample_rate` | `int` | `0` | Force playback rate (0 = auto/sticky) |
| `file_prefix` | `string` | `""` | Save each utterance as `<prefix>_NNNNN.wav` |
| `file_start_index` | `int` | `1` | First file index |
| `speaking_flag_delay` | `double` | `0.2` | s after playback until flag resets |
| `sentence_pause` | `double` | `0.6` | Silence after each played sentence (natural pacing) |
| `fade_ms` | `int` | `8` | Fade-in/out per utterance; removes stream-start clicks (`0` = off) |

## Emotion & Expression Tags (inline)

S2-Pro interprets bracket tags inside the text (`[tag] Beschreibung`) — 64+ documented
markers, and free-form natural-language descriptions work too
(e.g. `[slightly sad]`, `[mysterious][whispering]`). Tags are interpreted per synthesis
call: since each sentence is generated separately, put the tag **in front of every
sentence** it should style, or set `control_instructions` to apply it to everything.

**Shipped default: `[excited]`** — "energetic, enthusiastic" in the Fish S2 emotion
list — set in `config/container_params.yaml` (container), `config/s3ptts_params.yaml`
and the `s2pro.yaml` launch anchor. That is Eva's spritziger baseline; `warmup`
synthesis skips the preset. Live switch: `ros2 param set /eva/tts/tts
control_instructions '"[calm]"'` — the extra quoting matters, a bare `[calm]` is
parsed as a string array and rejected (`expected Type.STRING got Type.STRING_ARRAY`) —
neutral again with `""`, stronger with `[delighted]`, `[excited][confident]` or
free-form `[energetic, upbeat]`. Measured on one sentence (fixed `seed: 42`):
`""` → 89 frames / 4.13 s / rms 0.114, `[excited]` → 98 frames / 4.55 s / rms 0.123.

- **Emotions (49):** `happy` `sad` `angry` `excited` `calm` `nervous` `confident`
  `surprised` `scared` `worried` `frustrated` `empathetic` `embarrassed` `disgusted`
  `proud` `grateful` `curious` `sarcastic` `anxious` `hysterical` `nostalgic` `lonely`
  `hopeful` `determined` `regretful` `jealous` … (full list:
  [Fish Audio Emotion Reference](https://docs.fish.audio/developer-guide/core-features/emotions))
- **Tone (6):** `shouting` `screaming` `whispering` `soft tone` `in a hurry tone`,
  and word-level emphasis: `Das ist [emphasis] wichtig.`
- **Body sounds (11):** `laughing` `chuckling` `sobbing` `crying loudly` `sighing`
  `groaning` `panting` `gasping` `yawning` `snoring` `clear throat`
- **Ambient/pauses:** `audience laughing` `background laughter` `crowd laughing`,
  `[break]` (short) and `[long-break]` (long) pause.

**Bracket warning:** these tags live inside `[ ]`. If you configure `substitute`
filters (in the node or upstream in the launch), the regex character classes
**must not remove `[` or `]`** — a Markdown cleaner like `[*~_|<>\\^`\]\[]` turns
`[whisper]` into the *spoken word* " whisper ". Our defaults keep the brackets; only
`&nbsp;`, `<br>`, emoji, URLs, heading/`---` lines, backticks and other Markdown
punctuation get stripped.


## Voice (Eva) Cloning

S2-Pro voice cloning is in-context learning from 10–30 s of reference audio — Eva's 8.7 s
de / 6.0 s en references work but stay at the short end; record longer for better SIM.
The engine re-encodes the reference only when the file hash changes (`use_memory_cache: on`),
so swapping `voice_ref_audio`/`voice_ref_text` live via `ros2 param set` takes effect on the
next utterance.

**The reference clips live in this repo**: `config/eva_24khz_{de,en}.wav` and
`config/eva_44khz_{de,en}.wav` (same 8.7 s de / 6.0 s en takes, two sample rates). A clone
therefore builds a self-contained image, and since `docker-compose.yaml` bind-mounts
`./config` read-only, replacing a clip in the repo is picked up by the running container
after a restart — no rebuild.

## Hardware

S2-Pro: 4B slow-AR + 0.4B fast-AR backbone + 1.87 GB DAC codec, ~11 GB weights.
Out of the box fish-speech pre-allocates the KV cache for the model's full 32 k
context (`max_seq_len: 32768` in the snapshot `config.json`) → ~20.5 GB resident,
leaving almost no headroom; long utterances then die with CUDA OOM during DAC
decode. **We patch `text_config.max_seq_len` to 8192 in the local snapshot**
(prompt chunks + `max_new_tokens: 1024` fit inside it) and run with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in docker-compose.yaml).
After a fresh `hf download` re-apply the patch, or the node regrows to ~20 GB.

The model still needs the GPU **exclusively** —
any other model on the card (previous TTS containers, flux/moss image containers) must be
stopped first, otherwise the node dies with CUDA OOM while loading.

Measured latency: ~27 semantic tokens/s → 2–3 s utterances generated in 2.3–2.6 s
(`compile: true` is expected to improve this). Start flags that helped when close to the
limit: `warmup:=false` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## GUI

```bash
ros2 run bob_s2ptts gui
```

Real-time tuner talking to `/tts/set_parameters`, exposing the S2-Pro
parameter set (sampling, chunking, control tags, Eva reference, output).
