# Video Question Answering with Temporal Grounding

> Ask any question about a video and get a grounded answer with cited timestamps, plus four explainability views of the model's reasoning. Bilingual: English / العربية.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdelrahman-Nashaat/Video-Question-Answering-with-Temporal-Grounding/blob/main/video_qa_xai.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app)

This project tackles a single question: *given a video and a natural-language question about it, what is the answer and where in the video does that answer come from?* The system fuses two complementary modalities — what the video shows (SigLIP image embeddings of sampled frames) and what it says (Whisper transcript embedded with bge-m3) — to localise the most relevant moments, then synthesises a grounded answer via Groq's Llama 3.3 70B. Four orthogonal explainability views make the model's reasoning auditable, and a small evaluation suite reports temporal-grounding metrics over a held-out annotation set.

## Demo

<!-- screenshots go here -->

A short screen recording or screenshot of the Gradio app belongs here. When the notebook is run on Colab, `demo.launch(share=True)` returns a public URL valid for 72 hours that opens on any device.

## Key features

- **Multi-modal fusion.** Visual relevance (SigLIP) and audio-transcript relevance (Whisper + bge-m3) are scored independently, normalized via temperature-scaled z-softmax, and then linearly combined. Normalization is what makes the fused score actually discriminate between candidates instead of collapsing into a flat band.
- **Bilingual input.** Questions in English or Arabic both work end-to-end. SigLIP's text tower is English-only, so Arabic questions are translated for visual retrieval; the audio path uses bge-m3 directly (multilingual) and the LLM answers in the original language. Translations are surfaced in the Explainability tab so reviewers can see the system is transparent about it.
- **Four orthogonal XAI views.** Timeline relevance, Grad-CAM saliency on the top frame, a top-K candidate-frame grid with raw and normalized scores, and a stacked-bar modality-contribution chart that answers "did the answer come from sight or sound?".
- **Evaluation suite.** `run_evaluation` computes Top-1 IoU, Hit@K (K=1,3,5), and mean absolute temporal error over a user-supplied annotation set.
- **Auto-scaling encoders.** A runtime VRAM check picks SigLIP-large @ 384 and Whisper large-v3 on a T4 (>= 14 GB), or SigLIP-base @ 224 and Whisper medium on smaller GPUs / CPU.
- **Single-notebook deployment.** No separate server, no Docker, no Spaces config — open the notebook in Colab, run all, share the printed URL.

## How it works

The pipeline runs in a single Colab notebook. The video is downloaded (or read from an upload), and frames are sampled at 2 fps with a histogram-based deduplication step that drops near-duplicate consecutive frames so static shots collapse to one keyframe. The audio track is transcribed with `faster-whisper` (language is auto-detected per chunk), and the transcript is grouped into ~10-second sliding windows. Frames are embedded with SigLIP, transcript chunks with bge-m3 — both L2-normalized so cosine similarity is just a dot product. The question is embedded twice: with SigLIP for the visual side (after on-the-fly translation if Arabic) and with bge-m3 for the audio side (no translation — bge-m3 is multilingual). For each sampled timestamp the system computes a raw visual score (best similarity within ±1 s of the timestamp) and a raw audio score (best similarity over chunks containing the timestamp). Each modality's raw scores are then z-softmaxed at a low temperature, so the fused `α·v_norm + β·a_norm` score has visible peaks rather than the narrow band that raw cosine similarities tend to produce. The top-K combined moments (deduplicated so they don't overlap) are passed to Llama 3.3 70B via the Groq API, which writes a short answer in the same language as the question and cites the timestamps it used. Finally the four explainability views are rendered.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │  Inputs: video (file or URL) + question      │
                 └──────────────────────────┬───────────────────┘
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              ▼                                                           ▼
   ┌────────────────────┐                                    ┌──────────────────────┐
   │ Frame extraction   │                                    │ Audio extraction     │
   │  - 2 fps sampling  │                                    │  - faster-whisper    │
   │  - histogram dedup │                                    │    large-v3 (auto    │
   └─────────┬──────────┘                                    │    language detect)  │
             ▼                                               └─────────┬────────────┘
   ┌────────────────────┐                                              ▼
   │ SigLIP image embed │                                    ┌──────────────────────┐
   │ (large-384 / base) │                                    │ Sliding-window chunks│
   └─────────┬──────────┘                                    │  + bge-m3 embeddings │
             │                                               └─────────┬────────────┘
             └────────────────────────┬────────────────────────────────┘
                                      ▼
              ┌─────────────────────────────────────────────────┐
              │ Per-timestamp raw scores → z-softmax normalize  │
              │ → α·v_norm + β·a_norm → top-K dedup'd moments   │
              └─────────────────────────┬───────────────────────┘
                                        ▼
              ┌─────────────────────────────────────────────────┐
              │ Llama 3.3 70B (Groq) → grounded answer +        │
              │ cited [mm:ss] timestamps                        │
              └─────────────────────────┬───────────────────────┘
                                        ▼
              ┌─────────────────────────────────────────────────┐
              │ XAI: timeline · Grad-CAM · frame grid · modality│
              │      contribution                               │
              └─────────────────────────┬───────────────────────┘
                                        ▼
                                 ┌──────────────┐
                                 │  Gradio UI   │
                                 │  share=True  │
                                 └──────────────┘
```

Full per-stage rationale, alternatives considered, and known failure modes are in [`docs/architecture.md`](docs/architecture.md).

## Models

| Component | Model | Why |
|---|---|---|
| Visual encoder | `google/siglip-large-patch16-384` (or `siglip-base-patch16-224` on low-VRAM) | Sigmoid-loss training gives wider, more spread-out score distributions than CLIP, which makes peak detection in the timeline plot meaningful. |
| ASR | `faster-whisper` `large-v3` (or `medium`) | Strong multilingual recognition including Arabic dialects; CTranslate2 backend is ~4× faster than the reference Whisper. |
| Text retrieval | `BAAI/bge-m3` | Best open multilingual dense retriever; single 568M model embeds queries and chunks consistently. |
| Answer LLM | `llama-3.3-70b-versatile` (Groq) | Sub-second time-to-first-token, generous free tier, strong English & Arabic generation when grounded in retrieved evidence. |
| Explainability | `pytorch-grad-cam` + custom matplotlib plots | Standard, publication-friendly visualizations; Grad-CAM is computed on the same vision tower used for retrieval, so the saliency map is causally tied to the retrieval score. |

## Quick start (Google Colab)

1. Click the **Open in Colab** badge at the top of this README.
2. **Runtime → Change runtime type → T4 GPU.**
3. Get a free Groq API key at <https://console.groq.com/keys>.
4. In Colab, click the **🔑 Secrets** icon in the left sidebar, add a new secret named `GROQ_API_KEY` with your key, and toggle **Notebook access** on.
5. **Runtime → Run all.** First run downloads ~3 GB of model weights (a few minutes); subsequent runs reuse the Hugging Face cache.
6. The final cell prints a public URL like `https://xxxxx.gradio.live` — open it from any device.

## Local run (advanced)

A local run requires an NVIDIA GPU with ≥ 16 GB VRAM and `ffmpeg` on the system path. Set `GROQ_API_KEY` in a `.env` file (see `.env.example`), `pip install -r requirements.txt`, and open `video_qa_xai.ipynb` in Jupyter. Colab is the supported path.

## Evaluation

The notebook contains a `run_evaluation(eval_set)` function that runs the full pipeline against an annotated set and returns per-item metrics plus aggregate Hit@K bars. Each item is a dict:

```python
{"video": "/content/sample.mp4",
 "question": "What color is the cat?",
 "gt_timestamp": 12.0,
 "tolerance": 2.0}
```

Three metrics are reported:

- **Top-1 IoU** — temporal overlap between the predicted top moment's `±tolerance` window and the ground-truth `±tolerance` window.
- **Hit@K** (K = 1, 3, 5) — whether the ground-truth timestamp falls within `±3 s` of any of the top-K retrieved moments.
- **Mean absolute temporal error** — `|t_pred_top1 − t_gt|` averaged over items where a moment was returned.

Fill the `EVAL_SET` list in the evaluation cell, then run the next cell. Ground-truth labels for short clips can be produced quickly by scrubbing in any media player.

## Explainability outputs

### 1. Timeline relevance
Per-timestamp visual (blue) and audio (amber) raw cosine similarities, with the global mean drawn as a dotted line and the top-3 moments highlighted with vertical markers and `mm:ss` labels. Answers *when* relevance was concentrated.

### 2. Grad-CAM on the top frame
Saliency overlay on the highest-scoring frame using the same SigLIP vision tower that did the retrieval, so the heatmap is causally tied to the score. Answers *where in the frame* the model looked. Surfaces a status string (`Grad-CAM applied` or `Grad-CAM unavailable: …`) so failures are visible rather than silent.

### 3. Top candidate frames
Grid of the top-K frames with their raw V/A scores and the combined normalized score. The frame used in the answer gets a green border. Answers *what other moments were close runners-up*.

### 4. Modality contribution
Stacked bars per top moment showing the weighted normalized contribution of the visual vs audio modalities. Answers *did the model rely on what it saw or what it heard?* — at a glance.

## Limitations

- Designed for short videos (< 10 minutes). Longer videos will work but require lowering `FPS_SAMPLE` and raising `TRANSCRIPT_CHUNK_SECONDS`, and may exceed Colab's session memory or wall-clock limits.
- Whisper diarization is not used; multi-speaker accuracy depends entirely on the upstream ASR.
- SigLIP's text encoder is English-trained, so Arabic questions are translated to English for visual retrieval. The translation is shown in the UI for transparency. The audio path remains native-multilingual via bge-m3.
- URL downloads from YouTube, Instagram, and similar gated hosts often fail from Colab IPs (the Android/iOS player-client fallback helps for some YouTube videos but not all). Direct upload through the Gradio UI is the reliable path.
- The shared Gradio link expires after 72 hours; rerun the last cell to mint a new one.
- Highly abstract questions ("what is the mood of this video?") are out of scope — the system grounds answers in concrete frame and transcript evidence.

## Repository layout

```
.
├── video_qa_xai.ipynb     # Main notebook (40 cells: pipeline + evaluation + UI)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── docs/
│   └── architecture.md    # Detailed architecture, alternatives, failure modes
├── .env.example           # GROQ_API_KEY template (local runs)
└── .gitignore
```

## License

[MIT](https://opensource.org/licenses/MIT). Add a `LICENSE` file with the standard text if you fork or distribute.
