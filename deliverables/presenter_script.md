# Presenter Script — Video QA with Temporal Grounding

## Total target time: ~7 minutes

---

## Slide 1 — Title (15 seconds)

"Hello. This is my Computer Vision project — Video Question Answering with Temporal Grounding. I built a multi-modal system that takes any video and a free-form question, and returns a grounded answer with explainable reasoning. My name is Abdelrahman Nashaan, from Tanta University, Faculty of Artificial Intelligence."

*[click to advance]*

---

## Slide 2 — Problem (40 seconds)

"Let me start with the problem. Videos today are dense, time-consuming sources of information — people upload hours of content every minute, and most viewers only need one specific answer buried somewhere inside. Existing video tools fall into two camps: search engines that point you to the whole video without telling you when the answer appears, and generative models that produce a paragraph but cannot show you where they got it from. My goal was to bridge that gap — answer a free-form question about any video, and at the same time show exactly where in the timeline the answer lives and why the model believed it."

*[click to advance]*

---

## Slide 3 — Goals (30 seconds)

"So the project has four concrete goals. First, multi-modal — I fuse three signals: the visual frames, the audio transcript, and the question text. Second, bilingual — the system handles Arabic and English questions, and Arabic and English audio, end-to-end. Third, explainable — every answer comes with four orthogonal visualizations of the model's reasoning. And fourth, reproducible — the whole thing runs in a single notebook on free Google Colab GPU, so anyone can re-run my results without setting up infrastructure."

*[click to advance]*

---

## Slide 4 — Architecture (50 seconds)

"Here is the full pipeline. On the left side, frames are extracted at 2 frames per second with histogram-based deduplication so static shots collapse to one keyframe, and then embedded with SigLIP. On the right side, the audio track goes through Whisper large-v3 with auto language detection, gets chunked into 10-second sliding windows, and each chunk is embedded with bge-m3. Both streams meet at the scoring stage, where I compute per-timestamp visual and audio scores, normalize each modality with a temperature-scaled z-softmax, and combine them linearly to produce the top-K moments. Those moments and the matching transcript snippets are sent to Llama 3.3 70B on Groq, which writes the grounded answer with cited timestamps. Finally the four explainability views are rendered, and everything is wrapped in a Gradio interface."

*[click to advance]*

---

## Slide 5 — Multi-modal Pipeline (45 seconds)

"Let me unpack each stream in a bit more detail. The visual stream uses SigLIP, which I picked over CLIP because its sigmoid-loss training gives wider, more spread-out cosine similarity distributions — peaks become visible in the timeline plot instead of a flat band. The audio stream uses Whisper large-v3 because of its strong Arabic recognition, and bge-m3 to embed the transcript chunks because it is the strongest open multilingual dense retriever. The fusion step is a weighted sum of the normalized scores from both modalities — and the normalization is the single biggest quality lever in the whole pipeline. Finally the top-K moments are sent to Llama 3.3 on Groq for synthesis, because Groq gives me sub-second time-to-first-token and generous free quota."

*[click to advance]*

---

## Slide 6 — Bilingual Handling (40 seconds)

"One of the design constraints I gave myself was full Arabic and English support. Whisper large-v3 auto-detects the language of every audio segment, so code-switched videos are handled gracefully. bge-m3 is multilingual by construction, so Arabic transcripts are retrieved natively without any translation. The one place I had to compromise is SigLIP — its text tower is English-only, so for Arabic questions I auto-translate to English just for the visual retrieval side, and I show the translation to the user in the explainability tab so it is transparent rather than hidden. The final answer is always generated in the same language as the original question."

*[click to advance]*

---

## Slide 7 — Explainable AI (55 seconds)

"This is the part I am most proud of. I built four orthogonal explainability views that together answer four different questions. The timeline relevance plot shows visual and audio relevance over the full length of the video, with the top peaks marked — that answers *when*. The top candidate frame grid shows the K best frames side by side with their visual, audio, and combined scores — that answers *what other moments came close*. The modality contribution chart is a stacked bar per moment, decomposing whether the score came from sight or sound — that answers *which modality drove the decision*. And finally Grad-CAM is a spatial heatmap on the top frame, computed on the same SigLIP vision tower used for retrieval, so the saliency is causally tied to the score — that answers *where in the frame the model looked*. Together they answer what, where, when, and why."

*[click to advance]*

---

## Slide 8 — Evaluation (35 seconds)

"For evaluation I built a methodology, not a leaderboard claim. The user supplies annotated triples — a video, a question, and a ground-truth timestamp — and the system reports three metrics: Top-1 IoU, which measures temporal overlap between the predicted top moment and the ground truth; Hit@K for K equal to 1, 3, and 5, which measures whether the ground truth fell within any of the top-K retrieved moments; and Mean Temporal Error in seconds, which is the absolute distance from the predicted top moment to the ground truth. The annotated set is intentionally small — the contribution is the methodology and the reproducibility, not a benchmark number."

*[click to advance]*

---

## Slide 9 — Tech Stack (25 seconds)

"On the model side I use SigLIP for vision, Whisper large-v3 for ASR, bge-m3 for multilingual text retrieval, and Llama 3.3 70B on Groq for answer synthesis. On the tooling side it is PyTorch, Hugging Face Transformers, faster-whisper for the CTranslate2 backend, pytorch-grad-cam for the heatmaps, Gradio for the interface, the Groq API for the LLM, and Google Colab as the runtime. Every piece is open source or free-tier."

*[click to advance]*

---

## Slide 10 — Limitations & Future Work (40 seconds)

"I want to be honest about the limits. The system is tuned for short videos under ten minutes — longer videos still work but need config tuning. I do not use Whisper diarization, so multi-speaker accuracy depends on the upstream ASR alone. SigLIP is English-trained, which means Arabic visual queries inherit the translation quality. URL downloads from Colab IPs sometimes fail because of data-center blocks, so direct upload is the reliable path. As future work I would like to compare against an end-to-end video-language model, add speaker-aware retrieval through diarization, and push the temporal resolution finer than the current 2 fps sampling rate."

*[click to advance]*

---

## Live Demo (90 seconds)

*[switch to Colab tab]*

"Now let me show the system in action. I open the Colab notebook and scroll to the last cell, which has already been executed, and I click the public Gradio link it printed. *[click the share link]* The interface opens with two tabs — Answer and Explainability.

*[click the upload box]* I drop in a short sample video — let me wait a second for the upload to finish.

*[type into the question box]* Now I type a question — for example, *what color is the cat at the start of the video* — and click Run.

*[wait a few seconds for the run]* The Answer tab is now showing the LLM's response with the cited timestamp in `mm:ss` format, and the relevant transcript snippet underneath.

*[click the Explainability tab]* Switching to the Explainability tab — here is the timeline relevance plot, with the blue line being visual relevance and the amber line being audio relevance. You can see the peak right around the timestamp the LLM cited.

*[scroll down]* Below that is the top candidate frame grid — the chosen frame has a green border, and each thumbnail shows its visual, audio, and combined scores.

*[scroll down]* Next is the modality contribution chart — for this question the visual bar dominates, which is exactly what you would expect for a question about color.

*[scroll down]* And finally Grad-CAM on the top frame — you can see the heatmap concentrated on the cat, confirming the model attended to the right region. That is the full demo."

*[switch back to slides]*

---

## Slide 11 — Closing (15 seconds)

"That brings me to the end. The full code, the architecture document, and the evaluation suite are on GitHub at the link on screen. Thank you for listening — I am happy to take questions."

*[stop recording]*
