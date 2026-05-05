# Architecture

This document explains the design of the Video QA + Temporal Grounding system, the rationale for each component, and the trade-offs that were considered.

---

## 1. End-to-end diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUTS                                  │
│   Video (file or URL)  +  Question (AR or EN)                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────────────┐
│ Frame extract│   │ Audio extract +      │
│  (1 fps)     │   │ Whisper-large-v3     │
│              │   │ (word-level stamps)  │
└──────┬───────┘   └──────────┬───────────┘
       │                      │
       ▼                      ▼
┌──────────────┐   ┌──────────────────────┐
│ CLIP image   │   │ Transcript chunks    │
│ embeddings   │   │ (10s windows) → bge-m3│
│ (per frame)  │   │ embeddings           │
└──────┬───────┘   └──────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Multimodal Retrieval                        │
│  query → multilingual-CLIP text embed        │
│  query → bge-m3 text embed                   │
│  combined_score = α·visual + β·audio         │
│  → Top-K moments with (timestamp, scores)    │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  LLM Answer Synthesis                        │
│  Groq API (llama-3.3-70b-versatile)          │
│  Input: question + top-K moments + frame     │
│         captions + transcript snippets       │
│  Output: answer + cited timestamps           │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  XAI Visualizations                          │
│  1. Timeline plot (relevance over video)     │
│  2. Grad-CAM on top frame                    │
│  3. Frame-similarity heatmap (top-N frames)  │
│  4. Multimodal contribution chart            │
│     (visual vs audio score breakdown)        │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│  Gradio UI (with share=True)                 │
└─────────────────────────────────────────────┘
```

---

## 2. Per-stage justification

### 2.1 Frame extraction at 1 fps

**Choice.** Sample one frame per second with OpenCV.

**Why.** For the question types we target (object presence, scene changes, on-screen actions), neighbouring frames are nearly identical. 1 fps gives a 30× reduction in CLIP forward passes versus a 30 fps video at minimal recall cost. The retrieval window (`[t-1, t+1]`) further smooths over miss alignments.

### 2.2 CLIP-ViT-B/32 for visual embeddings

**Choice.** `openai/clip-vit-base-patch32` (base size, patch 32).

**Why.**
- Fits in T4 VRAM alongside Whisper-large-v3 and bge-m3.
- Has the strongest publicly reproduced zero-shot retrieval baseline at this size class.
- Compatible with `sentence-transformers/clip-ViT-B-32-multilingual-v1`, whose text encoder projects 50+ languages into the **same visual space**, enabling Arabic-→-image retrieval without translation.
- Compatible with `pytorch-grad-cam` for the XAI step.

A larger CLIP (`ViT-L/14`) would improve recall but breaks the multilingual text-encoder alignment and doubles VRAM usage.

### 2.3 Whisper-large-v3 via faster-whisper

**Choice.** `large-v3` with `compute_type="float16"` on the T4.

**Why.**
- Best Arabic ASR among open models (Whisper-v3 added significant Arabic data over v2).
- Auto-detects language **per segment**, which handles code-switched videos.
- `faster-whisper` (CTranslate2 backend) is ~4× faster than the reference implementation and fits large-v3 in ~3 GB VRAM in fp16.
- Returns word-level timestamps used to align retrieval windows precisely.

### 2.4 bge-m3 for transcript embeddings

**Choice.** `BAAI/bge-m3`, a multilingual dense + sparse retrieval model.

**Why.**
- 100+ languages including Arabic, with state-of-the-art MIRACL scores.
- Single 568M-parameter model embeds both **chunks** and **queries** consistently — no separate retrieval stack.
- Returns 1024-dim dense vectors that cosine-compare directly with the question embedding; we ignore the sparse and ColBERT-style outputs to keep the pipeline simple.

Alternatives considered: `intfloat/multilingual-e5-large` (similar size, slightly weaker on Arabic per the MIRACL leaderboard). For an English-only deployment, `bge-large-en-v1.5` would be marginally faster.

### 2.5 Multimodal score fusion

**Choice.** Linear combination `combined = α·visual + β·audio` with `α = β = 0.5` by default.

**Why.**
- Linear fusion is interpretable — the modality-contribution XAI chart is literally the per-term breakdown.
- `α` and `β` are exposed in the config; reviewers can rerun with `α=0`/`β=0` to see what each modality contributes alone.
- More elaborate late-fusion approaches (RRF, learned gating) were rejected because they complicate the explainability story without a clear quality win at this scale.

### 2.6 Groq llama-3.3-70b-versatile for answer synthesis

**Choice.** Groq-hosted llama-3.3-70b-versatile.

**Why.**
- **Free tier** with generous rate limits — fits the academic-demo budget.
- Sub-second time-to-first-token, which keeps the Gradio UX snappy.
- llama-3.3-70b handles Arabic generation well enough for short, grounded answers (we are not doing long-form Arabic composition, only summarising provided evidence).
- Uses retrieved evidence rather than internal knowledge → answers are auditable against the cited timestamps.

Alternatives considered: a local quantised LLM (Phi-3, Llama-3.1-8B-Instruct in 4-bit). Rejected because they would compete with CLIP/Whisper/bge-m3 for the T4's 16 GB VRAM.

### 2.7 Four XAI visualizations

**Choice.** Timeline, Grad-CAM, frame grid, modality-contribution.

**Why.** Each plot answers a distinct *type* of question a reviewer might ask:

| Plot | Question it answers |
|---|---|
| Timeline | *When* was relevance concentrated? |
| Grad-CAM | *Where in the frame* did the model look? |
| Frame grid | *Which other moments* were close runners-up? |
| Modality bars | *Which modality* drove the choice? |

Together they cover temporal, spatial, ranked, and modal explanation — a more complete picture than any single visualization.

---

## 3. Trade-offs vs alternatives

### 3.1 Why not Video-LLaVA / VideoChat / similar end-to-end VLMs?

End-to-end video LLMs would technically answer the question, but:
- They are **black boxes** — no native way to point to which seconds justified the answer, breaking the temporal-grounding requirement.
- Their explainability story is at best "attention heatmaps on a sampled frame," which is one of our four plots — not the whole pipeline.
- They are 7B-13B parameters of *new* compute on top of the embedding stack, blowing the T4 budget.

The retrieval-then-LLM design lets us ground answers in **concrete, auditable evidence** and decouple the four XAI views from the final language model.

### 3.2 Why not a vector database (FAISS, Chroma)?

For videos under ~30 minutes the number of frames (~1800) and chunks (~200) is small enough that brute-force cosine similarity in PyTorch is faster than the index-building overhead. Adding FAISS would also obscure the "every score is a dot product" simplicity that the modality-contribution plot relies on.

### 3.3 Why CLIP for both retrieval and Grad-CAM?

Reusing the same vision model for retrieval and saliency means the Grad-CAM heatmap is genuinely showing *what was matched*, not a separate model's interpretation. This is a meaningful explainability property: the heatmap is causally tied to the score.

### 3.4 Why a notebook instead of a packaged app?

The deliverable is academic, the runtime is Colab, and the user has no local GPU. A notebook:
- Is the standard Colab artifact (one click → run all → public URL).
- Keeps each pipeline stage in a clearly numbered, reviewable cell.
- Doubles as the report (markdown cells explain each step).

A standalone `app.py` can be extracted later if the project moves to Hugging Face Spaces.

---

## 4. Failure modes

| Scenario | Failure mode | Mitigation |
|---|---|---|
| Video > ~10 min | Frame embedding and transcription become slow (linear in length). | Lower `FPS_SAMPLE` to 0.5 and increase `TRANSCRIPT_CHUNK_SECONDS` to 20. |
| Silent video | Audio score is uniformly zero; retrieval falls back to visual only. | Detect zero-energy audio; in that case set `β = 0`. |
| Music-only audio | Whisper produces hallucinated transcripts. | The system flags `language="nan"` from Whisper and skips audio retrieval. |
| Very abstract question (mood, intent) | Retrieval surfaces unrelated moments; LLM may fabricate. | Out of scope — the system is for grounded, evidence-based questions. |
| Code-switched audio | Per-segment language detection prevents catastrophic failure but transcript quality varies. | Acceptable for the demo; future work could route Arabic vs English chunks to language-specific embedders. |
| Grad-CAM on Arabic question | CLIP text encoder is English-only. | Auto-translate the question via Groq for the saliency step only; show both versions in the UI. |
| Groq API outage / rate limit | Pipeline fails at the answer step. | Wrapped in try/except with a clear error message; the retrieval evidence is still shown so the user can read the answer themselves. |

---

## 5. Future work

- Replace the linear-fusion scorer with a small learned reranker on a held-out QA set.
- Add OCR (PaddleOCR or Tesseract) for slide-heavy videos so on-screen text becomes a third retrieval channel.
- Cache embeddings to disk keyed by video hash so the same video survives Colab session restarts.
- Add automatic chapter detection (PySceneDetect) to bias retrieval toward scene boundaries.
