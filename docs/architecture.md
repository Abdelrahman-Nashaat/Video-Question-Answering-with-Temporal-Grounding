# Architecture

This document describes the design of the Video QA + Temporal Grounding system, the rationale for each component, the alternatives considered, and the known failure modes.

---

## 1. End-to-end diagram

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
              │ Per-timestamp raw scores                        │
              │   visual: max similarity in ±1 s window         │
              │   audio:  max similarity over containing chunks │
              │ → temperature-scaled z-softmax per modality     │
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
              └─────────────────────────┬───────────────────────┘
                                        ▼
                                 ┌──────────────┐
                                 │  Gradio UI   │
                                 │  share=True  │
                                 └──────────────┘
```

---

## 2. Per-stage rationale

### 2.1 Frame extraction at 2 fps with histogram deduplication

Sampling at 1 fps misses fast cuts; sampling at 5+ fps wastes compute on near-duplicates. The pipeline samples at 2 fps and then drops any frame whose normalized 3-channel histogram correlates above 0.85 with its predecessor. Held shots collapse to a single representative frame, so per-second cost stays roughly constant whether the video is a slow interview or a fast-cut trailer.

### 2.2 SigLIP for visual embeddings

The visual encoder is `google/siglip-large-patch16-384` on a T4 (~3.4 GB), or `siglip-base-patch16-224` (~370 MB) on smaller GPUs. The choice is made at runtime by `detect_compute_tier()`.

SigLIP was preferred over CLIP for this task because its sigmoid-loss training produces wider, more spread-out cosine similarity distributions for the same query. With CLIP-base we observed top-K combined scores of 0.27 / 0.26 / 0.26 — visually identical bars in the modality-contribution plot and a flat timeline. SigLIP's wider spread, combined with the score-normalization step (next section), makes peak detection actually visible in the XAI plots.

The trade-off is that SigLIP's text tower is English-only, so Arabic queries require a translation step on the visual side (the audio side stays native via bge-m3).

### 2.3 Score normalization

Per-modality raw scores are passed through a temperature-scaled z-softmax:

```
z = (s − mean(s)) / std(s)
norm = softmax(z / T)        # T = 0.05 by default
```

This is the single biggest quality lever in the pipeline. Raw cosine similarities tend to live in a narrow band (e.g. 0.20 – 0.30 for SigLIP on natural questions), which means the top-K candidates are within rounding distance of each other — the timeline plot looks flat and the modality-contribution bars are indistinguishable. The z-softmax centres and rescales the distribution, then the low temperature sharpens it so genuine peaks dominate. Relative ordering is preserved.

The combined fused score is `α · v_norm + β · a_norm`. Because v_norm and a_norm are now both probability-mass distributions, the linear combination is interpretable as "fraction of the video's relevance budget at this timestamp."

### 2.4 Whisper-large-v3 via faster-whisper

ASR uses `large-v3` (or `medium` on low-VRAM tier) with `compute_type="float16"` on the T4. Whisper-v3 added significant Arabic data over v2 and is currently the strongest open ASR for the language. `faster-whisper`'s CTranslate2 backend is ~4× faster than the reference implementation. Word-level timestamps + per-segment language detection (which handles code-switched videos) are the features the rest of the pipeline depends on.

### 2.5 bge-m3 for transcript embeddings

`BAAI/bge-m3` is the strongest open multilingual dense retriever per MIRACL, supports 100+ languages including Arabic, and embeds both queries and chunks with the same 568M model. We use only the dense (1024-dim) output and ignore the sparse and ColBERT-style outputs to keep the pipeline simple.

### 2.6 Multimodal score fusion

`combined = α · v_norm + β · a_norm`, with `α = β = 0.5` by default. Linear fusion is the only choice that makes the modality-contribution chart literally a per-term decomposition. Both weights are exposed in `Config`; setting one to 0 isolates the other modality, which is useful for ablation.

### 2.7 Groq Llama 3.3 70B for answer synthesis

Groq is the answer LLM because of its low time-to-first-token, generous free tier, and strong Arabic + English performance on short grounded outputs. The model is constrained by a system prompt to answer only from the supplied evidence and to cite timestamps in `[mm:ss]` format.

A local quantised LLM (Phi-3, Llama 3.1 8B 4-bit) was rejected because it would compete with SigLIP, Whisper, and bge-m3 for the T4's 16 GB VRAM.

### 2.8 The four XAI views

Each plot answers a distinct kind of question:

| Plot | Question it answers |
|---|---|
| Timeline relevance | *When* in the video was relevance concentrated? |
| Grad-CAM on top frame | *Where in the frame* did the model look? |
| Frame-similarity grid | *What other moments* were close runners-up? |
| Modality contribution | *Which modality* drove the score? |

Together they cover temporal, spatial, ranked, and modal explanation. Grad-CAM specifically reuses the SigLIP vision tower that did retrieval, so the heatmap is causally tied to the score rather than coming from a separate explainer model.

---

## 3. Alternatives considered

### 3.1 Why not a video LLM (Video-LLaVA, VideoChat, etc.)?

End-to-end video LLMs would technically answer the question, but:

- They are black boxes — no native way to cite the seconds that justified the answer, breaking the temporal-grounding requirement.
- Their explainability is limited to attention-on-sampled-frames, which is one of our four plots, not the whole pipeline.
- They are 7B–13B parameters of new compute on top of the embedding stack; the T4 budget cannot afford that plus Whisper-large-v3 plus bge-m3.

The retrieval-then-LLM design grounds answers in concrete, auditable evidence and decouples the four XAI views from the final language model.

### 3.2 Why not a vector database?

For videos under ~30 minutes the number of frames (a few thousand at 2 fps) and chunks (a few hundred) is small enough that brute-force cosine similarity in PyTorch is faster than the index-building overhead. Adding FAISS or Chroma would also obscure the "every score is a dot product" simplicity that the modality-contribution plot relies on.

### 3.3 Why CLIP-style retrieval-then-Grad-CAM rather than VLM cross-attention?

Reusing the same vision tower for retrieval and saliency means the Grad-CAM heatmap shows what was *actually matched* during retrieval, not a separate model's interpretation. This is a meaningful explainability property: the heatmap is causally tied to the retrieval score.

### 3.4 Why a single notebook instead of a packaged app?

The deliverable is academic, the runtime is Colab, and the user has no local GPU. A notebook is the standard Colab artifact (open → run all → public URL), keeps each pipeline stage in a numbered cell that doubles as the report, and avoids any deployment friction. A standalone `app.py` could be extracted later for Hugging Face Spaces if the project moves there.

---

## 4. Failure modes

| Scenario | Failure mode | Mitigation |
|---|---|---|
| Video > ~10 min | Frame embedding and transcription become slow (linear in length). | Lower `FPS_SAMPLE` (e.g. 1.0), raise `TRANSCRIPT_CHUNK_SECONDS` (e.g. 20). |
| Silent video | Audio score is uniformly zero; retrieval falls back to visual only. | bge-m3 embeddings of an empty chunk list return an empty tensor; score normalization handles this without raising. |
| Music-only audio | Whisper produces hallucinated transcripts. | VAD filtering reduces this; the retrieval evidence in the answer prompt makes hallucinations easier for the LLM to ignore. |
| Highly abstract question (mood, intent) | Retrieval surfaces unrelated moments; LLM may fabricate. | Out of scope — the system is for grounded, evidence-based questions. |
| Code-switched audio | Per-segment language detection prevents catastrophic failure but transcript quality varies. | Acceptable for the demo; a future version could route Arabic vs English chunks to language-specific embedders. |
| Grad-CAM on Arabic question | SigLIP text tower is English-only. | The question is auto-translated via Groq for the saliency step; both versions are surfaced in the UI. |
| Grad-CAM internal failure | Older transformers / inference-mode loaders can break gradient flow. | The Grad-CAM call is wrapped — on failure the raw frame is returned with a status string explaining why. |
| Groq API outage / rate limit | Pipeline fails at the answer step. | Wrapped in try/except; the raw evidence is included in the failure message so the answer can be read manually. |
| YouTube / Instagram URL fails to download | Data-center IP block or login wall. | yt-dlp tries Android/iOS player clients first for YouTube; on failure a `VideoDownloadError` carries an actionable hint and the Gradio UI surfaces it cleanly. Direct upload always works. |

---

## 5. Future work

- Replace the linear-fusion scorer with a small learned reranker on a held-out QA set.
- Add OCR (PaddleOCR or Tesseract) for slide-heavy videos so on-screen text becomes a third retrieval channel.
- Persist embedding cache to disk keyed by video hash so the same video survives Colab session restarts.
- Add automatic chapter detection (PySceneDetect) to bias retrieval toward scene boundaries.
- Add diarization to support speaker-targeted questions ("what did the second speaker say at...").
