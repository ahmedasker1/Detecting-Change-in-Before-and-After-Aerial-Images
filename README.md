# Aerial Image Change Detection (LEVIR-CD) — Siamese U-Net

مشروع كامل لاكتشاف التغيّر بين صور جوية ملتقطة في وقتين مختلفين (bi-temporal
change detection) باستخدام مجموعة بيانات **LEVIR-CD** ونموذج **Siamese U-Net**
مبني من الصفر بـ PyTorch.

يحلّ النموذج المسألة التالية: تُعطى صورتان (زمن 1 `A` وزمن 2 `B`) لنفس
الموقع، ويخرج النموذج قناعًا ثنائيًا (binary mask) يحدّد البيكسلات التي
تغيّرت (بناء/هدم مبانٍ عادةً).

---

## 1) بنية المشروع

```
DEPI Project/
├── explore_data.py        # فحص الداتا: الشجرة، المطابقة، الأحجام، القيم، عينات بصرية
├── train.py               # التدريب + early stopping + حفظ أفضل checkpoint (+ --resume)
├── evaluate.py            # تقييم test: مقاييس كاملة + عتبة auto + confusion + رسوم
├── make_notebooks.py      # مولّد دفاتر Jupyter (يمكن إعادة توليدها في أي وقت)
├── smoke_test.py          # اختبار سريع لخط الأنابيب ببيانات اصطناعية
├── diagnose.py            # أداة تشخيص (eval vs train BN، جودة العتبات)
├── requirements.txt
├── README.md
├── notebooks/             # دفتر لكل مرحلة (kernel: Python (levir-cd))
│   ├── 00_setup.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── config.py          # كل الإعدادات في مكان واحد (seed، image size، batch، lr، ...)
│   ├── data_discovery.py  # اكتشاف بنية الداتا تلقائيًا ومطابقة A/B/mask بالأسماء المشتركة
│   ├── dataset.py         # Datasets + augmentations متزامنة (flips/rot90) + تقسيم إلى tiles
│   ├── model.py           # Siamese U-Net (encoder مشترك الأوزان + دمج بالتسلسل + decoder)
│   ├── losses.py          # BCEWithLogitsLoss + Dice Loss (مناسب لعدم توازن الفئات)
│   └── metrics.py         # مقاييس لكل فئة + mIoU + Balanced Accuracy + Kappa
└── outputs/               # يُنشأ تلقائيًا: checkpoints، منحنيات، تقارير JSON، صور
    └── run_YYYYmmdd_HHMMSS/   # كل تشغيل تدريب في مجلد مستقل (لا يُكتب فوقه أبدًا)
```

## 2) التثبيت

```bash
# بيئة افتراضية (يُنصح بـ Python 3.10–3.11)
conda create -n levir-cd python=3.11 -y
conda activate levir-cd

# PyTorch مع CUDA (اختر النسخة المناسبة لتعريف كرت الشاشة)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# باقي الاعتماديات
pip install -r requirements.txt
```

على جهاز بدون GPU يكفي `pip install torch` وسيستخدم الكود CPU تلقائيًا.

## 3) التشغيل — مسارَان: Jupyter أو Python

### المسار التفاعلي (Jupyter) — دفتر لكل مرحلة

```bash
D:\envs\levir-cd\python.exe -m jupyter lab notebooks
```
اختر الـkernel **Python (levir-cd)** ثم نفّذ الدفاتر بالترتيب:

| الدفتر | الغرض |
|---|---|
| `notebooks/00_setup.ipynb` | فحص البيئة والـGPU |
| `notebooks/01_data_exploration.ipynb` | تنزيل الداتا، شجرة المجلدات، المطابقة، الإحصاءات، عينات بصرية، توزيع نسبة التغيّر |
| `notebooks/02_training.ipynb` | بناء النموذج + حلقة تدريب تفاعلية + منحنيات (كل تحسن يُحفظ فورًا — المقاطعة لا تفقد الأفضل) |
| `notebooks/03_evaluation.ipynb` | ضبط العتبة على validation + تقرير test الكامل بمقاييس غير مضللة + confusion matrix + أمثلة بصرية |

الدفاتر تستدعي نفس محرك `src/` الذي تستخدمه السكربتات — لا يوجد منطق مكرر.

### المسار غير التفاعلي (Python)

```bash
python explore_data.py                  # فحص الداتا + عينات + تقرير JSON
python train.py                         # تدريب كامل (كل التشغيل في outputs/run_<timestamp>/)
python train.py --epochs 60 --batch-size 16 --lr 1e-3 --image-size 256
python train.py --resume outputs\run_XXXX\best_model.pt --epochs 25 --lr 2.5e-4   # استكمال
python evaluate.py --run-dir outputs\run_XXXX --threshold auto
```
`--threshold auto` يجرب العتبات 0.30–0.70 على **validation** ويستخدم الأفضل
(أعلى F1) على test — مطبوع ومحفوظ في `test_metrics.json`. وبدون `--run-dir`
يختار السكربت تلقائيًا الـrun صاحب أعلى val IoU (يطبع المرشحين قبل الاختيار)،
حتى لا يلتقط تشغيلات تجريبية قصيرة مثل `run_nb_*` من دفتر التدريب.

## 4) طريقة الحساب والتعريفات (مقاييس غير مضللة)

- التنبؤ: `pred = sigmoid(logits) > threshold` (0.5 افتراضيًا، أو ضبط تلقائي
  على validation بـ`--threshold auto`).
- **مقاييس فئة التغيّر (headline):** `Precision = TP/(TP+FP)`،
  `Recall = TP/(TP+FN)`، `F1 = 2PR/(P+R)`، `IoU = TP/(TP+FP+FN)` —
  على مستوى البكسل عبر عدّاد confusion واحد يتراكم على كامل الـtest set.
- **مقاييس مجمّعة مقاومة لعدم التوازن:** `Mean IoU` (متوسط IoU الفئتين)،
  `Balanced Accuracy` (متوسط recall الفئتين)، و`Cohen's Kappa`
  ((p_o − p_e)/(1 − p_e)) الذي يصحح الاتفاق العشوائي.
- **جدول لكل فئة** (change / unchanged) بنفس التعريفات مع تبديل أدوار TP/TN،
  وconfusion matrix بصيغتين: نسبة من كل البكسل ونسبة صفية طبيعية.
- تنبيه: Accuracy ≈ 97.7% تبدو مرتفعة لكنها فقاعة — 95.8% من البكسل غير
  متغيرة أصلًا، لذا لا تُعتمد إلا كرقم مرجعي بجوار المقاييس أعلاه.

## 5) الافتراضات والقرارات (مهم)

- **بنية الداتا الفعلية (مكتشفة تلقائيًا)**: الداتا المنزّلة هي **LEVIR-CD+**
  ببنية `LEVIR-CD+/train/{A,B,label}` (637 صورة) و `LEVIR-CD+/test/{A,B,label}`
  (348 صورة)، كل صورة **1024×1024 RGB (PNG)**، والأقنعة ثنائية بقيم {0,1}.
- لا يوجد مجلد `val` رسمي، لذلك يطبّق الكود قاعدة واضحة: عند غياب/نقص الـsplits
  الرسمية تُجمَع كل العينات (985) ويُعاد تقسيمها **70/15/15 بـseed=42** مع فحص
  تسريب (لا عينة في أكثر من split): train=689، val=147، test=149.
- **اكتشاف المجلدات تلقائيًا** ولا يُفترض مسبقًا: الكود يمشي في الشجرة، يصنّف
  المجلدات بكلمات مفتاحية (A/B, image1/image2, mask/label, ...)، ويطابق الملفات
  بجذور الأسماء المشتركة (file stem). إن فشل الاكتشاف يتوقف برسالة واضحة ويطلب
  تعبئة `OVERRIDES` في `src/data_discovery.py`.
- الصور (1024×1024) أكبر من `IMAGE_SIZE` (256): التدريب بقصّات عشوائية 256×256
  متزامنة على الصورتين والقناع، والتقييم بشبكة tiles كاملة 16×16 لكل صورة ثم
  تُعاد تجميع التنبؤات لصورة كاملة.
- الـaugmentations هندسية فقط (انعكاس أفقي/رأسي، دوران بـ k×90°) وتُطبَّق بنفس
  المعاملات على الصورتين والقناع معًا — لا تكسر التطابق المكاني ولا تفقد ثنائية القناع.
- normalize للصور: `/255` ثم `(x-0.5)/0.5`؛ القناع يبقى {0,1} دائمًا، ويُقرأ
  بشكل متسامح (0/1 أو 0/255 أو 0/128) عبر عتبة عند نصف القيمة القصوى.
- بيئة التشغيل الفعلية: conda env باسم `levir-cd` على `D:\envs\levir-cd`
  (Python 3.11 + PyTorch 2.14.1+cu126) — القرص C: كان شبه ممتلئ فنُقلت البيئة
  والكاشات إلى D:. استخدم `D:\envs\levir-cd\python.exe` لتشغيل السكربتات.
  كما يعمل `--num-workers 0` افتراضيًا هنا لأن فتح عمليات فرعية يعيد تحميل
  DLLات CUDA ويسبب "paging file too small" على هذا الجهاز.

## 6) الأجهزة

- استُخدم في التشغيل الفعلي: RTX 3050 Laptop (4 GB) مع mixed precision،
  `base_channels=32`، `batch_size=16` عند 256×256، `--num-workers 0`.
- بدون GPU يعمل الكود على CPU تلقائيًا (أبطأ فقط).

## 7) ما تم تشغيله فعليًا والنتائج النهائية

جميع الأرقام التالية من تشغيل حقيقي على هذا الجهاز (RTX 3050 Laptop 4GB،
بيئة `D:\envs\levir-cd`، PyTorch 2.14.1+cu126):

- `explore_data.py` ✔ — التقرير في `outputs/data_report.json` والعينات في
  `outputs/figures/data_samples.png` (985 عينة 1024×1024، أقنعة {0,1}، ~4.2–4.9%
  بكسل متغيرة لكل split).
- **التدريب (مرحلتان):** ✔
  1. `run_20261009_200513`: 40 epoch من الصفر → best val IoU = 0.576.
  2. `run_20261011_073203`: استكمال 25 epoch من أفضل checkpoint
     (`--resume ... --lr 2.5e-4`) → **best val IoU = 0.610** (epoch 18).
- `evaluate.py --threshold auto` ✔ — ضبط العتبة على validation (اختار 0.30،
  أعلى val F1=0.761) ثم تقييم مرة واحدة على test (149 صورة = 2384 بلاطة
  256×256). النتائج في `outputs/run_20261011_073203/eval_20261011_082622/`:

  | المقياس | القيمة |
  |---|---|
  | Precision (change) | 0.7336 |
  | Recall (change) | 0.7841 |
  | **F1 (change)** | **0.7580** |
  | **IoU (change)** | **0.6103** |
  | Mean IoU (فئتان) | 0.7939 |
  | Balanced Accuracy | 0.8856 |
  | Cohen's Kappa | 0.7467 |
  | Overall Accuracy | 0.9783 (غير معبّرة: 95.7% من البكسل غير متغيرة) |

  Per-class: change (P .734 / R .784 / F1 .758 / IoU .610) —
  unchanged (P .990 / R .987 / F1 .989 / IoU .978).
  Confusion (بكسل): TP=5.31M، FP=1.93M، FN=1.46M، TN=147.53M —
  الصورة في `confusion_matrix.png` (بصيغتين: نسبة كلية ونسبة صفية)، و6 أمثلة
  بصرية في `predictions/`، والملف الكامل (مع جدول مسح العتبات على validation)
  في `test_metrics.json`.

ملاحظة صادقة عن التشخيص أثناء التطوير: أول تشغيلين توقفا مبكرًا بسبب خطأ
قياس (double-sigmoid في حساب مقاييس الـvalidation — النموذج كان يتعلم جيدًا
لكن المقاييس كانت متجمدة)، تم إصلاحه ثم إعادة التدريب من الصفر؛ النتائج
أعلاه من التشغيلات النظيفة. للتحسين لاحقًا: تدريب أطول/أعرض، أو TTA، أو
استبدال BatchNorm بـGroupNorm لتقليل التذبذب بين epochs.
