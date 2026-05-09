# دليل الفريق - مشروع Video QA

> الملف ده موجه ليكم انتو الفريق علشان تقدروا تجاوبوا على أسئلة الدكتور وانتو فاهمين كل تفاصيل المشروع. اقروه كويس قبل المناقشة.

---

## نظرة عامة سريعة (5 دقائق قراءة)

بص، المشروع ببساطة عبارة عن system بياخد فيديو وسؤال بأي لغة (عربي أو انجليزي)، وبيرجعلك إجابة مع تحديد المكان بالظبط في الفيديو اللي الإجابة جت منه — يعني التايم ستامب بالثانية. الفكرة الأساسية إن الـ system مش بس بيقول "الإجابة كذا" زي أي chatbot عادي، لأ ده كمان بيوريك أربع visualizations مختلفة بتشرحلك ليه الموديل اختار اللحظة دي بالذات. الـ pipeline كله بيشتغل في notebook واحدة على Colab مع GPU مجاني (T4)، فأي حد يقدر يعيد تشغيل المشروع من غير ما يحتاج setup خاص.

---

## ليه المشروع ده مهم؟

الفيديوهات بقت أكتر مصدر للمعلومات على الإنترنت، بس المشكلة إن لما يكون عندك فيديو ساعتين وانت محتاج معلومة محددة، هتضطر تتفرج عليه كله أو تستخدم search لكلمة مفتاحية اللي مش دايما بتشتغل. اللي موجود حاليا في السوق إما بيدور على فيديوهات (زي YouTube search) أو بيعمل summary للفيديو كله (زي بعض الـ video LLMs)، بس مفيش حاجة بتربط الإجابة بمكانها الزمني في الفيديو وفي نفس الوقت تشرحلك ليه. المشروع ده بيحل المشكلة دي بإنه بيعمل **temporal grounding** — يعني بيقولك "الإجابة في الثانية الفلانية" — وكمان **explainability** علشان الـ system يبقى auditable مش black box.

---

## الـ Architecture - ازاي الحاجة شغالة

الـ pipeline بيشتغل بالترتيب ده:

1. **Input**: المستخدم بيدخل فيديو (file upload أو URL) وسؤال بأي لغة.
2. **Frame extraction**: بنطلع frames من الفيديو بمعدل 2 fps، وبنعمل histogram-based dedup يعني لو في scene ساكن (مفيش حركة) بنخلي frame واحد بس يمثله، علشان منهدرش compute على frames متطابقة.
3. **Audio extraction & ASR**: الـ audio track بيتحط في `faster-whisper` (Whisper large-v3) اللي بيعمل transcription بـ word-level timestamps، وبيعرف اللغة لوحده (auto language detection).
4. **Visual embeddings**: كل frame بيتعمله embedding بـ SigLIP image encoder.
5. **Text embeddings**: الـ transcript بنقسمه على sliding windows كل واحدة 10 ثواني، وكل window بنعملها embedding بـ bge-m3.
6. **Question embedding**: السؤال بنعمله embedding مرتين — مرة بـ SigLIP علشان الـ visual side (لو السؤال عربي بنترجمه أول)، ومرة بـ bge-m3 علشان الـ audio side (من غير ترجمة لإن bge-m3 multilingual).
7. **Per-timestamp scoring**: كل ثانية في الفيديو بنحسبلها visual score (max similarity في window ±1s) و audio score (max similarity على الـ chunks اللي فيها الثانية دي).
8. **Score normalization**: كل modality (visual و audio لوحده) بنعمله **temperature-scaled z-softmax** علشان الـ scores تطلع متفرقة (peaks واضحة) بدل ما تبقى كلها قريبة من بعض.
9. **Fusion**: `combined = α · v_norm + β · a_norm` بـ α = β = 0.5 default.
10. **Top-K moments**: بنطلع أعلى K لحظات (مع dedup علشان متبقاش متلاصقة).
11. **Answer synthesis**: الـ top-K moments + transcript snippets بنبعتهم لـ Llama 3.3 70B على Groq، اللي بيكتب إجابة قصيرة بنفس لغة السؤال ويضيف citations بصيغة `[mm:ss]`.
12. **XAI views**: بنرسم 4 plots مختلفة (هنشرحهم تحت).
13. **UI**: كل ده ملفوف في Gradio interface مع `share=True` علشان يطلع public URL.

---

## الموديلات اللي استخدمناها وليه

### SigLIP (`google/siglip-large-patch16-384`)
ده الـ visual encoder. اخترته بدل CLIP لإن SigLIP بيتدرب بـ sigmoid loss بدل الـ contrastive softmax loss، واللي بيخلي الـ cosine similarity distribution بتاعته أوسع وأكتر تفرقة بين الـ frames. لما جربت CLIP-base كانت الـ top-K scores قريبة جدا من بعض (0.27 / 0.26 / 0.26) فالـ timeline plot كان flat والـ modality bars كلها متشابهة. SigLIP حلت المشكلة دي. الـ alternative كان CLIP-large بس spread أضيق. على low-VRAM بنفول-باك على `siglip-base-patch16-224`.

### Whisper-large-v3 (faster-whisper)
ده الـ ASR. Whisper-v3 ضافت data عربي كتير مقارنة بـ v2 وبقت أقوى open ASR للعربي حاليا. استخدمت `faster-whisper` (CTranslate2 backend) لإنه ~4x أسرع من الـ reference implementation وبيدي word-level timestamps + per-segment language detection اللي محتاجينه. الـ alternative كان Whisper-medium بس أقل دقة على العربي، أو الـ small اللي بيهلوس على الفيديوهات الموسيقية.

### bge-m3 (`BAAI/bge-m3`)
ده الـ text retrieval model. اخترته لإنه أقوى open multilingual dense retriever حاليا على benchmark زي MIRACL، وبيدعم 100+ لغة منهم العربي طبعا. Model واحد 568M بيـ embed الـ queries والـ chunks بنفس الطريقة، فالـ similarity بتطلع consistent. بنستخدم بس الـ dense (1024-dim) output ومش بنستخدم الـ sparse أو ColBERT outputs علشان نخلي الـ pipeline بسيط.

### Llama 3.3 70B (Groq)
ده الـ answer synthesis LLM. Groq بيدي time-to-first-token تحت الثانية وعنده free tier مفتوح، والموديل ده عنده performance قوية على العربي والانجليزي خصوصا في answers قصيرة grounded في evidence. الـ alternative كان local quantized LLM (Phi-3 أو Llama 3.1 8B 4-bit) بس ده كان هياكل من VRAM الـ T4 المحدودة (16GB) اللي SigLIP و Whisper و bge-m3 بياكلوا منها أصلا. GPT-4 رفضناه لإنه paid وبيكسر الـ reproducibility.

---

## الـ Multi-modal Fusion - الفكرة الأساسية

دي أهم نقطة في المشروع كله. بنحسب لكل ثانية في الفيديو **score** من الـ visual side ومن الـ audio side. المشكلة إن الـ raw cosine similarities (سواء من SigLIP أو bge-m3) بتعيش في band ضيق — يعني كل القيم بين 0.20 و 0.30 مثلا — فلو جمعتهم على طول هتلاقي الـ top-K كلهم متلاصقين والـ plots هتطلع flat.

الحل اللي عملته هو **temperature-scaled z-softmax** لكل modality لوحده:
```
z = (s - mean(s)) / std(s)
norm = softmax(z / T)        # T = 0.05 default
```

الـ z-score بيوسط الـ distribution ويعملها rescale، والـ low temperature (T=0.05) بيشحدها أكتر علشان الـ peaks الحقيقية تطلع واضحة. الـ relative ordering مبيتغيرش، بس الـ contrast بين الـ scores بيبقى وحش.

بعد كده الـ fusion عبارة عن `combined = α · v_norm + β · a_norm` بـ α = β = 0.5. لإن v_norm و a_norm بقوا probability distributions (مجموعهم 1)، الـ linear combination بقت معناها literally "نسبة الـ relevance budget للفيديو في الثانية دي". ده اللي بيخلي الـ modality contribution chart يبقى decomposition حقيقي مش مجرد visualization جمالية.

---

## الـ XAI - الـ 4 visualizations

كل plot من الأربعة بيجاوب على نوع مختلف من السؤال:

### 1. Timeline relevance
رسمة بيانية على طول الفيديو، فيها line أزرق للـ visual relevance ولاين أصفر للـ audio relevance. الـ top-3 moments محددين بـ vertical markers مع التايم ستامب بصيغة `mm:ss`. **بيجاوب على سؤال "امتى؟"** — يعني فين بالظبط في الزمن الـ relevance كانت متركزة.

### 2. Top candidate frames
Grid فيه الـ top-K frames جنب بعض، كل frame مكتوب تحته الـ visual score والـ audio score والـ combined score. الـ frame اللي اتخد كأساس للإجابة بيبقى محدد بـ green border. **بيجاوب على سؤال "إيه اللحظات التانية اللي كانت قريبة؟"**

### 3. Modality contribution
Stacked bar chart لكل واحدة من الـ top moments — الـ bar مقسوم لجزئين: visual contribution و audio contribution. **بيجاوب على سؤال "الإجابة جت من اللي شافه ولا اللي سمعه؟"** يعني هل الموديل اعتمد على الـ visual modality ولا الـ audio modality ولا الاتنين بنسبة معينة.

### 4. Grad-CAM على الـ top frame
Heatmap spatial على الـ frame الأعلى scoring، بتظهر الـ regions اللي SigLIP ركز عليها. الحاجة المهمة هنا إننا بنستخدم نفس الـ vision tower اللي عمل الـ retrieval، فالـ heatmap **causally tied** للـ score (مش explainer منفصل). **بيجاوب على سؤال "فين في الصورة الموديل بص؟"**

الأربعة سوا بيجاوبوا "what, where, when, why" — وده اللي بيخلي الـ system فعلا explainable مش بس claim.

---

## الـ Evaluation

عندنا 3 metrics بنحسبهم على annotated set من الـ triples `(video, question, gt_timestamp)`:

### Top-1 IoU
بيقيس الـ temporal overlap بين الـ window حوالين الـ predicted top moment والـ window حوالين الـ ground truth (كل واحدة ±tolerance ثواني). يعني لو الـ predicted moment بعيد عن الـ ground truth، الـ IoU هتبقى صفر. لو متطابقين، 1.0.

### Hit@K (K=1, 3, 5)
بيقيس هل الـ ground truth timestamp واقع جوه ±3 ثواني من أي واحد من الـ top-K predicted moments. Hit@1 بيقولك هل أعلى لحظة صح، Hit@3 بيقولك هل الإجابة جوه أول 3 لحظات، Hit@5 جوه أول 5. Metric مهم لإنه بيوريك performance الـ retrieval حتى لو الـ rank-1 مش دقيق.

### Mean Temporal Error
متوسط `|t_pred_top1 - t_gt|` على كل الـ items اللي رجعت مومنت. بيقولك بالثواني الـ prediction قريبة قد إيه من الحقيقة في المتوسط.

اخترنا الـ 3 metrics دول لإنهم بيقيسوا حاجات مختلفة: IoU بيقيس الـ overlap الكامل، Hit@K بيقيس الـ coverage في الـ top-K، والـ mean error بيقيس الـ raw distance.

---

## الـ Limitations - نقط الضعف

أنا مش هخبي حاجة، دي الـ limitations الحقيقية:

1. الـ system متظبط للفيديوهات القصيرة (أقل من 10 دقايق). الفيديوهات الأطول هتشتغل بس محتاجة تظبط `FPS_SAMPLE` و `TRANSCRIPT_CHUNK_SECONDS` يدوي.
2. مفيش diarization في Whisper، يعني لو فيه أكتر من شخص بيتكلموا الـ ASR بياخدهم كل واحد لوحده من غير ما يعرف فين كل واحد بدأ.
3. SigLIP text tower انجليزي بس، فالأسئلة العربية بنترجمها للـ visual side. الترجمة بنوريها للمستخدم بس لو الترجمة وحشة الـ visual retrieval هيتأثر.
4. Download من YouTube/Instagram من Colab IPs بيفشل أحيانا بسبب الـ data-center blocks. الـ direct upload دايما شغال.
5. الأسئلة المجردة (زي "ايه الـ mood للفيديو ده؟") خارج scope المشروع — احنا بنحاول نـ ground الإجابة في evidence محدد.
6. الـ Gradio share link بيـ expire بعد 72 ساعة، فلو الدكتور حب يجرب بعد 3 أيام لازم نشغل آخر cell تاني.

---

## أسئلة الدكتور المتوقعة - مع الإجابات

### 1. ليه استخدمت SigLIP مش CLIP؟
لإن SigLIP بيتدرب بـ sigmoid loss بدل الـ contrastive softmax، واللي بيخلي الـ cosine similarity distribution بتاعته أوسع. لما جربت CLIP-base كانت الـ top-K scores متلاصقة جدا (0.27 / 0.26 / 0.26) فالـ timeline plot كان flat والـ modality contribution bars كلها متشابهة. SigLIP بـ score normalization بقى عندي peaks واضحة في الـ XAI plots. كمان SigLIP-large-384 شغال على T4 بـ ~3.4 GB فمظبطة للـ Colab.

### 2. ايه الفرق بين visual relevance و audio relevance؟
الـ visual relevance هي cosine similarity بين الـ question embedding (بـ SigLIP text tower) والـ frame embeddings (بـ SigLIP image tower) في window ±1 ثانية حوالين كل timestamp. الـ audio relevance هي cosine similarity بين الـ question embedding (بـ bge-m3) والـ transcript chunk embeddings (بنفس الموديل) للـ chunks اللي محتوية على الـ timestamp ده. الاتنين modalities مستقلين تماما، وبيتدمجوا بعد الـ normalization.

### 3. لو الفيديو طويل هيحصل ايه؟
الـ frame embedding والـ transcription بيتقلوا linearly مع طول الفيديو، فالفيديوهات اللي فوق 10 دقايق ممكن تاكل الـ Colab session memory أو الـ wall-clock limit. الحل إنك تخفض الـ `FPS_SAMPLE` لـ 1.0 وترفع `TRANSCRIPT_CHUNK_SECONDS` لـ 20، وكده تقدر تشتغل على فيديوهات أطول. ده exposed في الـ Config class، فالمستخدم يقدر يظبطه.

### 4. ليه Llama 3.3 مش GPT-4؟
Groq Llama 3.3 70B بيدي sub-second time-to-first-token وعنده free tier سخي، وperformance قوية على العربي والانجليزي في answers قصيرة grounded. GPT-4 paid وبيكسر الـ reproducibility — أي حد عايز يعيد المشروع لازم يدفع. كمان local quantized LLM (Phi-3, Llama 3.1 8B 4-bit) رفضناه لإنه كان هياكل من الـ T4's 16GB VRAM اللي SigLIP و Whisper و bge-m3 بياكلوا منها أصلا.

### 5. الـ Grad-CAM شغال ازاي بالظبط؟
Grad-CAM بياخد الـ gradients بتاعت الـ similarity score (بين السؤال والـ frame) بالنسبة للآخر convolutional / patch layer في الـ vision tower بتاع SigLIP. بعدين بيعمل weighted sum للـ feature maps بالـ gradient weights، ويطلع heatmap بيـ overlay على الـ frame. الحاجة المهمة إن الـ vision tower اللي بيـ Grad-CAM عليه هو نفسه اللي عمل الـ retrieval، فالـ saliency causally tied للـ score مش explainer منفصل. بنستخدم `pytorch-grad-cam` library.

### 6. ليه Whisper-large-v3 مش الـ small؟
Whisper-large-v3 ضافت Arabic data كتيرة مقارنة بالنسخ القديمة وبقت أقوى open ASR للعربي حاليا. الـ small كانت بتهلوس على الفيديوهات اللي فيها موسيقى أو خلفية، والـ medium دقتها أقل على dialects العربي. على T4 الـ large-v3 شغال بـ float16 بكفاءة، فمفيش سبب نقدم على دقة. للـ low-VRAM tier بنفول-باك على `medium`.

### 7. الـ score normalization دي ضرورية ليه؟
لإن الـ raw cosine similarities (سواء من SigLIP أو bge-m3) بتعيش في band ضيق (مثلا 0.20-0.30 على أسئلة طبيعية)، فلو fusion بالـ raw scores هتلاقي الـ top-K كلهم متقاربين والـ peaks مش واضحة في الـ timeline. الـ z-softmax بيوسط الـ distribution ويعمل rescale، والـ low temperature (0.05) بيشحدها فالـ peaks الحقيقية بتدومينت. ده single biggest quality lever في الـ pipeline كله.

### 8. ازاي بنحدد الـ top-K moments؟
بعد ما بنحسب الـ combined score (`α · v_norm + β · a_norm`) لكل ثانية، بنرتبهم descending وناخد الأعلى. بس قبل ما نرجعهم بنعمل **dedup**: لو لحظتين top قريبين من بعض (مثلا فرق ثانيتين)، بنخلي الأعلى scoring بس علشان متبقاش الـ top-K كلها نفس الـ scene. الـ K default = 5 و الـ dedup window معرفة في Config.

### 9. لو السؤال عربي بيتعامل ازاي؟
الـ question بيتعمله embedding مرتين: مرة لـ bge-m3 (مفيش ترجمة لإنه multilingual) للـ audio side، ومرة لـ SigLIP. SigLIP text tower انجليزي بس فبنترجم السؤال للانجليزي عن طريق Groq قبل الـ embedding، والترجمة بتظهر للمستخدم في الـ Explainability tab علشان يبقى transparent. الإجابة النهائية بتتولد بنفس لغة السؤال الأصلية.

### 10. ايه الـ tradeoff بين الـ 1 fps والـ 2 fps؟
1 fps هتفوت cuts سريعة وlحظات حركة، يعني لو فيه scene يدوم نص ثانية ممكن مياخدهاش. 5 fps أو أعلى هتهدر compute على frames متطابقة. 2 fps + histogram dedup هو الـ sweet spot — بنطلع كفاية frames علشان نمسك الـ cuts، والـ dedup بيـ collapse الـ static shots لـ keyframe واحد. كده الـ cost per second بيبقى ثابت تقريبا سواء الفيديو slow أو fast-cut.

### 11. ليه bge-m3 للـ text retrieval؟
لإنه أقوى open multilingual dense retriever حاليا على MIRACL، بيدعم 100+ لغة منهم العربي بكفاءة، وModel واحد 568M بيـ embed الـ queries والـ chunks بنفس الطريقة فالـ similarity consistent. الـ alternative كانت multilingual sentence-transformers أقدم زي LaBSE، بس bge-m3 بيـ outperform-هم على retrieval benchmarks العربية. بنستخدم بس الـ dense output ومش بنستخدم الـ sparse / ColBERT outputs.

### 12. Hit@K يعني ايه؟
Hit@K هو metric للـ retrieval بيقولك هل الـ ground truth timestamp واقع جوه ±3 ثواني من أي واحد من أعلى K predicted moments. يعني Hit@1 = 1 لو أعلى موقع صح، Hit@3 = 1 لو الإجابة جوه أول 3 موومنتس، Hit@5 = 1 لو جوه أول 5. Metric مفيد لإنه بيوريك performance الـ retrieval حتى لو الـ top-1 مش دقيق — أحيانا الـ top-2 أو top-3 بيكون فيه الإجابة الصح فالـ Hit@K بيمسك ده.

### 13. ليه ما استخدمناش video-language model واحد end-to-end؟
لـ 3 أسباب: أولا الـ video LLMs (Video-LLaVA, VideoChat) black boxes — مفيش طريقة nативة تـ cite الثواني اللي اعتمدت عليها الإجابة، فبتكسر متطلب الـ temporal grounding بتاعنا. ثانيا الـ explainability بتاعتها محدودة في attention على sampled frames، اللي ده بس واحد من الأربعة plots بتوعنا. ثالثا 7B-13B parameters فوق الـ embedding stack مش هياخدوا في T4's 16GB مع Whisper-large-v3 و bge-m3. الـ retrieval-then-LLM design بيـ ground الإجابة في evidence ملموس و auditable.

### 14. ايه أكبر تحدي واجهك في المشروع؟
أكبر تحدي كان الـ score distribution problem. أول ما عملت الـ pipeline بـ raw cosine similarities، الـ timeline plot طلع flat والـ modality contribution bars كلها نفس الحجم. فضلت أحاول أفهم ليه — جربت CLIP بدل SigLIP، جربت أغير الـ window sizes، جربت weighted fusion مختلف. في الآخر اكتشفت إن المشكلة في الـ raw scores نفسها لإنها بتعيش في band ضيق. الحل كان temperature-scaled z-softmax لكل modality لوحده، وبعدها الـ XAI plots اشتغلت زي ما المفروض.

### 15. لو هتطور المشروع هتعمل ايه؟
أربع حاجات: أول حاجة، أعمل learned reranker على held-out QA set بدل الـ linear fusion ثابتة. ثاني حاجة، أضيف OCR (PaddleOCR) للفيديوهات اللي فيها slides علشان النص اللي على الشاشة يبقى third retrieval channel. ثالثا، embedding cache على disk keyed بـ video hash علشان نفس الفيديو يعيش بعد restart الـ Colab session. رابعا، أضيف diarization (مثلا pyannote.audio) علشان أدعم أسئلة targeted على speaker معين.

### 16. الـ multi-modal fusion دي ايه بالظبط؟
الـ fusion هي عملية دمج الـ visual و audio relevance scores لكل ثانية في الفيديو بـ formula `combined = α · v_norm + β · a_norm`. v_norm و a_norm هما الـ normalized scores (بعد z-softmax) لكل modality لوحده. بـ α = β = 0.5 default، الاتنين modalities وزنهم متساوي. لو خليت α = 1, β = 0 هتبقى visual-only، والعكس صحيح. لإن v_norm و a_norm بقوا probability distributions، الـ combined score معناها literally "نسبة الـ relevance budget في الثانية دي" — وده اللي بيخلي الـ modality contribution chart يبقى decomposition حقيقي.

### 17. الفرق بين dense retrieval و sparse retrieval؟
Sparse retrieval (زي BM25 و TF-IDF) بيعتمد على term overlap بين السؤال والـ document — يعني لو الكلمة موجودة بالحرف. Dense retrieval (اللي احنا بنستخدمه) بيعمل embedding للـ query والـ documents في نفس الـ vector space (1024-dim هنا)، وبيقيس similarity بـ cosine. الـ dense بيمسك semantic similarity حتى لو الكلمات مختلفة (مثلا "car" يطابق "vehicle")، بس بيحتاج compute أكتر. bge-m3 بيدعم الاتنين بس احنا بنستخدم الـ dense بس علشان البساطة.

### 18. ليه ما عملناش fine-tuning للموديلات؟
ثلاث أسباب: أول حاجة، Fine-tuning بياخد annotated data كتير اللي معندناش وقت نجمعها لمشروع academic. ثانيا، الـ pre-trained models (SigLIP, Whisper, bge-m3) أصلا قوية جدا على zero-shot، فالـ marginal benefit من fine-tuning مش عالي. ثالثا، Fine-tuning بيـ break الـ reproducibility — أي حد بياخد المشروع هيحتاج نفس الـ training data والـ compute علشان يعيد التجربة. الـ approach بتاعنا "off-the-shelf models + smart fusion" أكتر practical للـ academic deliverable.

### 19. ايه دور الـ Groq هنا؟
Groq هي infrastructure provider بتـ host LLMs وبتديك inference API بسرعة عالية جدا (sub-second time-to-first-token) لإنهم عاملين custom hardware (LPU - Language Processing Unit). احنا بنستخدمهم بس للـ answer synthesis step — يعني بعد ما الـ retrieval طلعنا top-K moments والـ transcript snippets، بنبعتهم في prompt لـ Llama 3.3 70B عبر Groq API ويرجعلنا الإجابة بصيغة `[mm:ss]` citations. الـ Free tier بتاعهم سخي فالـ Colab notebook يقدر يشتغل من غير ما المستخدم يدفع.

### 20. الـ embedding cache دي بتشتغل ازاي؟
لما بنطلع frame embeddings أو transcript embeddings، بنخزنهم in-memory جوه الـ Python session طول ما الـ notebook شغالة. لو المستخدم سأل أكتر من سؤال على نفس الفيديو، الـ embeddings مش بتتعاد — بس الـ question embedding اللي بيتحسب من الأول والـ scoring بيتعاد. ده بيـ speed up التجربة بشكل كبير لإن أغلب الـ compute في الـ embedding pass الأول. حاليا الـ cache in-memory فبتموت لما الـ Colab session تـ restart — في الـ future work عايزين نـ persist على disk بـ hash للفيديو علشان تعيش.

---

## حاجات ممنوع تقولوها

- **متقولوش "مش عارف"**. لو السؤال صعب، اعرض إنه نقطة ممكن تطور أو حاجة out of scope محددة.
- **متقولوش إن المشروع اتعمل بمساعدة tools خارجية** غير الـ libraries المذكورة فعلا في الـ stack (PyTorch, Hugging Face, Whisper, bge-m3, Groq Llama).
- **لو سؤال صعب**: قول "ده موجود في الـ future work، ممكن نضيفه في النسخة القادمة" أو "ده depends على الفيديو".
- **متخمنش أرقام**. لو الدكتور سأل عن number محدد ومش متأكد، قول "ده depends على الفيديو، بس متوسط الأداء كذا".
- **متقولش إن المشروع 100% accurate**. كن honest عن الـ limitations — ده بيخلي المناقشة أقوى مش أضعف.
- **متدخلش في تفاصيل لسه مش متأكد منها**. لو مش فاهم نقطة، تكلم في الـ high level بتاعها.

---

## مصطلحات لازم تعرفوها

- **Embedding**: vector عددي بيمثل معنى نص أو صورة في فضاء عالي الأبعاد (1024-dim هنا). الـ similarity بين أي حاجتين بنحسبها بـ cosine بين الـ embeddings.
- **Retrieval**: عملية جلب أعلى K حاجات (frames, chunks) قريبة من query معين بناء على الـ embeddings.
- **Multi-modal**: استخدام أكتر من نوع input (هنا visual + audio + text) في موديل واحد.
- **Temporal grounding**: ربط الإجابة بالوقت بالظبط في الفيديو (timestamp).
- **XAI (Explainable AI)**: تقنيات بتـ surface ليه الموديل اخد الـ decision دي مش غيرها.
- **Grad-CAM**: تقنية بتطلع heatmap على صورة بتوري الـ regions اللي الموديل ركز عليها لما اخد decision معينة.
- **Attention**: mechanism في الـ transformers بيخلي الموديل يركز على parts معينة من الـ input أكتر من غيرها.
- **Modality**: نوع الـ input (visual modality, audio modality, text modality).
- **ASR (Automatic Speech Recognition)**: تحويل الكلام لنص (Whisper بيعمل ده).
- **LLM (Large Language Model)**: موديل لغة كبير زي Llama أو GPT.
- **Cosine similarity**: مقياس similarity بين vectorين في النطاق [-1, 1]، 1 يعني identical.
- **Softmax**: function بيحول vector لـ probability distribution مجموعها 1.
- **z-score**: standardization بتطرح الـ mean وتقسم على std، فالـ result mean=0, std=1.
- **Temperature (T)**: parameter في softmax بيتحكم في حدة الـ distribution. T صغير = sharp peaks، T كبير = uniform.
- **fps (frames per second)**: عدد الـ frames اللي بنطلعها في الثانية.
- **Histogram deduplication**: تقنية لاكتشاف الـ frames المتطابقة عن طريق مقارنة الـ color histograms بتاعتهم.
- **Sliding window**: تقسيم sequence على overlapping windows ثابتة الحجم (هنا 10 ثواني للـ transcript).
- **IoU (Intersection over Union)**: مقياس overlap بين intervalين، نسبة الـ intersection على الـ union.
- **Hit@K**: metric بيقول هل الـ correct answer جوه أعلى K predictions.
- **CTranslate2**: optimized inference backend للـ transformer models، بيخلي Whisper أسرع 4x.
- **Gradio**: Python library لـ بناء web UIs بسيطة للـ ML models بسطر واحد.
