# Video Question Answering with Temporal Grounding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdelrahman-Nashaat/Video-Question-Answering-with-Temporal-Grounding/blob/main/video_qa_xai.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app)

> Ask any question about a video — in **Arabic or English** — and get an answer **with the exact moments** in the video where the answer comes from, plus visual explanations of why the model picked those moments.

> اطرح أي سؤال حول فيديو — **بالعربية أو الإنجليزية** — واحصل على الإجابة مع **اللحظات الزمنية** التي استُمدت منها، إلى جانب تفسيرات بصرية لقرار النموذج.

---

## Demo

> *Add a screenshot or GIF of the Gradio UI here once captured.*

When the notebook is run on Colab, `gradio.launch(share=True)` provides a **public URL valid for 72 hours** that can be opened from any device.

---

## Features

- **Multi-modal retrieval** — fuses what the model **sees** (CLIP frame embeddings) with what it **hears** (Whisper transcript embeddings).
- **Temporal grounding** — every answer cites the **exact timestamps** in the video that justify it.
- **Bilingual end-to-end** — accepts questions and audio in **Arabic or English**, no manual translation required.
- **Four complementary XAI visualizations** — timeline relevance plot, Grad-CAM on the top frame, frame-similarity grid, and a modality-contribution chart that shows whether the answer came from sight or sound.
- **No local GPU needed** — runs entirely on Google Colab (free T4 tier) and exposes a shareable Gradio link.
- **Embedding cache** — follow-up questions on the same video are answered instantly.

---

## Architecture

```
Inputs ──► Frame extraction (1 fps) ──► CLIP image embeddings ──┐
                                                                ├──► Multimodal scorer ──► Top-K moments ──► Groq LLM ──► Answer + timestamps
       └──► Audio extraction ──► Whisper-large-v3 ──► bge-m3 ───┘                                                          │
                                                                                                                            ▼
                                                                                            XAI: timeline · Grad-CAM · frame grid · modality chart
                                                                                                                            ▼
                                                                                                                       Gradio UI
```

A more detailed diagram and design rationale lives in [`docs/architecture.md`](docs/architecture.md).

---

## How it works

1. **Ingest** — accepts an uploaded file or a URL (YouTube / direct mp4). `yt-dlp` downloads it locally.
2. **Frame sampling** — `OpenCV` extracts one frame per second; each frame is embedded with `CLIP-ViT-B/32`.
3. **Audio transcription** — `faster-whisper` (`large-v3`) produces word-level timestamps and auto-detects the language.
4. **Transcript indexing** — segments are merged into ~10 s windows and embedded with `BAAI/bge-m3` (multilingual).
5. **Retrieval** — the question is embedded twice: with multilingual-CLIP (for visual matching) and with bge-m3 (for transcript matching). A weighted score `α·visual + β·audio` ranks every timestamp and returns the top-K moments.
6. **Answer synthesis** — the question, top moments, and their transcripts are sent to Groq's `llama-3.3-70b-versatile`, which answers in the same language as the question and cites the timestamps.
7. **Explanation** — four plots are generated to make the model's reasoning auditable.

---

## Models used

| Model | Role | Source |
|---|---|---|
| `openai/clip-vit-base-patch32` | Frame visual embeddings | Hugging Face |
| `sentence-transformers/clip-ViT-B-32-multilingual-v1` | Multilingual text → CLIP space (Arabic/English questions) | Hugging Face |
| `BAAI/bge-m3` | Multilingual transcript & question embeddings | Hugging Face |
| `large-v3` (faster-whisper) | Speech-to-text with timestamps, language auto-detect | Systran / OpenAI |
| `llama-3.3-70b-versatile` | Final answer generation | Groq API |
| `pytorch-grad-cam` | Visual saliency on top frame | jacobgil/pytorch-grad-cam |

---

## Run on Google Colab

1. Click the **Open in Colab** badge at the top of this README.
2. **Runtime → Change runtime type → T4 GPU** (free tier is sufficient).
3. Get a free Groq API key from <https://console.groq.com/keys>.
4. In Colab: **🔑 Secrets panel → Add new secret → Name: `GROQ_API_KEY`** → paste your key → toggle **Notebook access** on.
5. **Runtime → Run all.** First run downloads ~3 GB of model weights (a few minutes). Subsequent runs reuse the cache.
6. The last cell prints a public Gradio URL like `https://xxxxx.gradio.live` — open it on any device for the demo.

---

## Local run

> Not the primary path. Requires an NVIDIA GPU with ≥10 GB VRAM and `ffmpeg` installed.

```bash
git clone https://github.com/Abdelrahman-Nashaat/Video-Question-Answering-with-Temporal-Grounding.git
cd Video-Question-Answering-with-Temporal-Grounding
cp .env.example .env  # then edit GROQ_API_KEY
pip install -r requirements.txt
jupyter notebook video_qa_xai.ipynb
```

---

## XAI components

Each of the four visualizations answers a different question about the model's reasoning:

| Visualization | Question it answers |
|---|---|
| **Timeline relevance plot** | *Where in the video was relevance concentrated, and was it driven by what was seen or heard?* |
| **Grad-CAM on top frame** | *Within the most relevant frame, which pixels matched the question?* |
| **Frame-similarity grid** | *What were the model's other candidate moments, and how close were they?* |
| **Modality contribution chart** | *For each top moment, did the visual or the audio modality drive the score?* |

Together they let a reviewer audit both the **temporal** decision (which seconds) and the **modal** decision (sight vs sound).

---

## Limitations

- Best on videos under ~5 minutes; longer videos work but increase indexing time linearly. Tune `FPS_SAMPLE` and `TRANSCRIPT_CHUNK_SECONDS` in the config cell to trade fidelity for speed.
- Silent videos fall back to visual-only retrieval; questions about spoken content will fail.
- Highly abstract questions ("what is the mood of this video?") are out of scope — the system grounds answers in concrete frame and transcript evidence.
- Grad-CAM requires English text aligned with CLIP; Arabic questions are auto-translated via Groq for the saliency step only (both versions are shown in the UI for transparency).
- The shared Gradio link is valid for **72 hours**; rerun the last cell to regenerate.

---

## Acknowledgments

- OpenAI for [CLIP](https://github.com/openai/CLIP) and [Whisper](https://github.com/openai/whisper).
- Systran for [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- BAAI for [bge-m3](https://huggingface.co/BAAI/bge-m3).
- Sentence-Transformers for the [multilingual CLIP](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) projection.
- Jacob Gildenblat for [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
- [Groq](https://groq.com) for fast LLM inference.
- [Gradio](https://gradio.app) for the UI.

---

## License

MIT — see [LICENSE](LICENSE) (add the file if you fork).
