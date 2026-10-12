# -*- coding: utf-8 -*-
"""Generate the Jupyter notebooks (one per project phase) into notebooks/.

Run:  D:\\envs\\levir-cd\\python.exe make_notebooks.py

The notebooks are thin, interactive wrappers around the same `src/` engine the
.py entry points use -- one notebook per phase:
  00_setup.ipynb            environment / GPU check
  01_data_exploration.ipynb download, tree, matching, stats, visual samples
  02_training.ipynb         datasets -> model -> training loop -> curves
  03_evaluation.ipynb       threshold tuning + full test report + figures
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(True)}


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (levir-cd)",
                           "language": "python", "name": "levir-cd"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }


PY_PATH_SETUP = (
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "PROJECT_ROOT = Path.cwd()\n"
    "if not (PROJECT_ROOT / \"src\").exists():   # kernel cwd may be notebooks/\n"
    "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
    "sys.path.insert(0, str(PROJECT_ROOT))\n"
)


# ===========================================================================
SETUP = nb([
    md("""# 00 — تجهيز البيئة (LEVIR-CD Change Detection)

هذا الدفتر يتحقق من أن بيئة التشغيل جاهزة: Python / PyTorch / GPU.
هذه **بيئة** التشغيل الفعلية على هذا الجهاز: conda env باسم `levir-cd`
على `D:\\envs\\levir-cd` (Python 3.11 + PyTorch cu126).

إعدادات التثبيت من سطر الأوامر (مرة واحدة فقط):

```bash
conda create -p D:\\envs\\levir-cd python=3.11 -y
D:\\envs\\levir-cd\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
D:\\envs\\levir-cd\\python.exe -m pip install -r requirements.txt
D:\\envs\\levir-cd\\python.exe -m ipykernel install --user --name levir-cd --display-name "Python (levir-cd)"
```

ثم اختر الـkernel باسم **Python (levir-cd)** عند فتح أي دفتر.

> **ملاحظة تشغيل على هذا الجهاز:** نستخدم `num_workers=0` في DataLoader
> لأن فتح عمليات فرعية على Windows يعيد تحميل مكتبات CUDA ويستهلك الذاكرة
> الافتراضية (paging file) بشكل يسبب فشلًا على أجهزة ذات قرص C: ممتلئ."""),
    code("""
import sys
import torch
import numpy
import matplotlib
import PIL

print("python    :", sys.version.split()[0])
print("torch     :", torch.__version__)
print("numpy     :", numpy.__version__)
print("matplotlib:", matplotlib.__version__)
print("CUDA      :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU       :", torch.cuda.get_device_name(0))
    print("VRAM      :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""),
])


# ===========================================================================
EXPLORATION = nb([
    md("""# 01 — فحص وتجهيز الداتا (LEVIR-CD+)

الهدف: تنزيل الداتا عبر `kagglehub`، اكتشاف البنية **بدون افتراض أسماء المجلدات**،
مطابقة صور الزمنين والأقنعة بالمعرف المشترك، فحص الأبعاد/القنوات/القيم،
وإحصاءات التقسيم 70/15/15، ثم عرض عينات بصرية.

المنطق الفعلي موجود في `src/data_discovery.py` و`src/config.py` — هذا الدفتر
واجهة تفاعلية له. (البديل غير التفاعلي: `python explore_data.py`)"""),
    code(PY_PATH_SETUP + """
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config
from src.data_discovery import (build_samples, change_ratio_stats, download_dataset,
                                load_mask, print_tree, split_report, verify_samples)

config.ensure_dirs()
print("project root:", PROJECT_ROOT)
"""),
    code("""
root = download_dataset()
print_tree(root)
"""),
    md("""## اكتشاف المجلدات ومطابقة العينات

التصنيف يعتمد كلمات مفتاحية (A/B, image1/image2, mask/label ...) ويطابق الملفات
بجذور الأسماء (file stem). إن فشل الاكتشاف توقف الكود برسالة واضحة، ويمكن
تعبئة `OVERRIDES` في `src/data_discovery.py` يدويًا."""),
    code("""
samples, notes = build_samples(root)
print(f"Total matched samples: {len(samples)}")
for s in samples[:5]:
    print("  e.g.", s.stem, "|", s.split, "|", s.a_path.name, s.b_path.name, s.mask_path.name)
print("\\nAssumptions / notes:")
for n in notes:
    print("  -", n)
"""),
    code("""
ver = verify_samples(samples)
print("image sizes :", ver["sizes"])
print("image modes :", ver["image_modes"])
print("mask values :", ver["mask_unique_values"])
print("problems    :", ver["problems"] or "none")
"""),
    code("""
report = split_report(samples)
print(f"{'split':>6} | {'n':>5} | {'changed%':>9} | {'unchanged%':>10} | empty masks")
for split, st in report.items():
    print(f"{split:>6} | {st['n_samples']:5d} | {st['changed_frac_mean']*100:8.2f}% | "
          f"{st['unchanged_frac_mean']*100:9.2f}% | {st['empty_masks']}/{st['n_masks']}")
"""),
    md("## عينات بصرية"),
    code("""
rng = np.random.default_rng(config.SEED)
train_samples = [s for s in samples if s.split == "train"]
picks = rng.choice(len(train_samples), size=3, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(13, 13))
for row, idx in enumerate(picks):
    s = train_samples[int(idx)]
    a = np.array(Image.open(s.a_path).convert("RGB"))
    b = np.array(Image.open(s.b_path).convert("RGB"))
    m = load_mask(s.mask_path)
    axes[row, 0].imshow(a); axes[row, 0].set_title(f"{s.stem} | t1 (A)")
    axes[row, 1].imshow(b); axes[row, 1].set_title("t2 (B)")
    axes[row, 2].imshow(m, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title(f"change mask ({m.mean()*100:.1f}%)")
    for c in range(3):
        axes[row, c].axis("off")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "data_samples.png", dpi=110)
plt.show()
"""),
    code("""
# توزيع نسبة التغيّر لكل عينة (على عينة من 200 قناع)
r = [load_mask(s.mask_path).mean() for s in train_samples[:200]]
plt.figure(figsize=(7, 3.5))
plt.hist(np.array(r) * 100, bins=30, color="#3b6ea5")
plt.xlabel("changed pixels per image (%)")
plt.ylabel("images")
plt.title("Distribution of change ratio (train sample)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""),
])


# ===========================================================================
TRAINING = nb([
    md("""# 02 — التدريب: Siamese U-Net لاكتشاف التغيّر

- **المعمارية:** encoder مشترك الأوزان (Siamese) يعالج الصورتين، ثم دمج
  بالتسلسل (concatenation) عند كل مستوى، وdecoder بنمط U-Net يخرج logits
  لقناع التغيّر.
- **الخسارة:** `BCEWithLogitsLoss + Dice` (مناسبة لعدم توازن ~5% بكسل متغير).
- **Augmentation متزامن:** قصّة عشوائية 256×256 + انعكاس أفقي/رأسي + دوران
  k×90° بنفس المعاملات على (A, B, mask) — لا يكسر التطابق المكاني.
- **المقاييس لكل epoch:** loss / Precision / Recall / F1 / IoU على train وval،
  مع early stopping وحفظ أفضل checkpoint حسب val IoU.

> التدريب الكامل (~40 epoch) يستغرق ≈ 1.5–2 ساعة على RTX 3050. لتنفيذ كامل
> غير تفاعلي استخدم `python train.py`. هذا الدفتر ينفّذ نفس الحلقة تفاعليًا،
> ويمكنك مقاطعته (interrupt) دون فقدان أفضل checkpoint لأنه يُحفظ في كل تحسن.

**تنبيه مهم:** العداد `ConfusionMeter` يطبّق sigmoid داخليًا — يجب تمرير
**logits خام** وليس احتمالات (تمرير probs يسبب double-sigmoid ويفسد المقاييس؛
خطأ اكتُشف وأُصلح في هذا المشروع)."""),
    code(PY_PATH_SETUP + """
import random, time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data_discovery import build_samples, download_dataset
from src.dataset import EvalTileDataset, TrainChangeDataset, samples_for_split
from src.losses import BCEDiceLoss
from src.metrics import ConfusionMeter
from src.model import SiameseUNet, count_parameters

SEED = config.SEED
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print("device:", device, "| AMP:", USE_AMP)
"""),
    code("""
root = download_dataset()
samples, notes = build_samples(root)
train_s = samples_for_split(samples, "train")
val_s = samples_for_split(samples, "val")
print(f"train={len(train_s)}  val={len(val_s)}")
for n in notes:
    print("  note:", n)

train_ds = TrainChangeDataset(train_s, image_size=config.IMAGE_SIZE, augment=True)
val_ds = EvalTileDataset(val_s, image_size=config.IMAGE_SIZE)  # tiles حتمية للـvalidation
train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                      num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                    num_workers=config.NUM_WORKERS, pin_memory=False)
print(f"train steps/epoch: {len(train_dl)}  |  val tiles: {len(val_ds)}")
"""),
    code("""
model = SiameseUNet(in_ch=config.IN_CHANNELS, base=config.BASE_CHANNELS).to(device)
print(f"parameters: {count_parameters(model)/1e6:.2f} M")

criterion = BCEDiceLoss(config.BCE_WEIGHT, config.DICE_WEIGHT).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR,
    patience=config.LR_SCHEDULER_PATIENCE)
scaler = torch.amp.GradScaler(enabled=USE_AMP)

run_dir = config.OUTPUT_DIR / f"run_nb_{time.strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
print("run dir:", run_dir)
"""),
    code("""
def run_epoch(model, loader, criterion, optimizer, scaler, device, threshold=0.5):
    model.train()
    meter, losses = ConfusionMeter(), []
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(a, b)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(a, b)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
        losses.append(loss.item())
        meter.update(logits.float(), y, threshold)   # <-- logits خام
    return float(np.mean(losses)), meter.metrics()


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    meter, losses = ConfusionMeter(), []
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        if USE_AMP:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(a, b); loss = criterion(logits, y)
        else:
            logits = model(a, b); loss = criterion(logits, y)
        losses.append(loss.item())
        metas = batch["meta"]
        for j in range(logits.shape[0]):   # استبعاد الـpadding من المقاييس
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            meter.update(logits.float()[j, :, :vh, :vw], y[j, :, :vh, :vw], threshold)
    return float(np.mean(losses)), meter.metrics()
print("epoch functions ready")
"""),
    md("""## حلقة التدريب

اضبط `EPOCHS` (الافتراضي هنا 5 كتجربة سريعة؛ التدريب الكامل 40 عبر `train.py`
أو ضع 40 هنا واستعد للانتظار). الـcheckpoint الأفضل يُحفظ تلقائيًا عند كل تحسن."""),
    code("""
EPOCHS = 5            # quick demo; full training: python train.py or EPOCHS = 40
history, best_iou, best_epoch = [], -1.0, -1

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_m = run_epoch(model, train_dl, criterion, optimizer, scaler, device)
    va_loss, va_m = eval_epoch(model, val_dl, criterion, device)
    scheduler.step(va_m["iou"])
    history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                    "train_iou": tr_m["iou"], "val_iou": va_m["iou"],
                    "val_f1": va_m["f1"], "lr": optimizer.param_groups[0]["lr"]})
    star = ""
    if va_m["iou"] > best_iou:
        best_iou, best_epoch = va_m["iou"], epoch
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "val_iou": va_m["iou"], "val_f1": va_m["f1"]},
                   run_dir / "best_model.pt")
        star = "  *best*"
    print(f"ep {epoch:3d} | train {tr_loss:.4f} IoU {tr_m['iou']:.3f} | "
          f"val {va_loss:.4f} IoU {va_m['iou']:.3f} F1 {va_m['f1']:.3f} "
          f"({time.time()-t0:.0f}s){star}")
print(f"\\nbest epoch {best_epoch} | best val IoU {best_iou:.4f}")
print("checkpoint:", run_dir / "best_model.pt")
"""),
    code("""
ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
ax[0].set_title("Loss (BCE+Dice)"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ep, [h["train_iou"] for h in history], label="train IoU")
ax[1].plot(ep, [h["val_iou"] for h in history], label="val IoU")
ax[1].set_title("IoU"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[2].plot(ep, [h["val_f1"] for h in history], label="val F1")
ax[2].set_title("F1 (change)"); ax[2].legend(); ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(run_dir / "curves.png", dpi=110)
plt.show()
"""),
])


# ===========================================================================
EVALUATION = nb([
    md("""# 03 — التقييم على Test set (بمقاييس غير مضللة)

**لماذا Accuracy مضللة هنا؟** في LEVIR-CD نحو 95% من البكسل غير متغيرة؛ نموذج
يصنف كل شيء "بدون تغيير" يحقق Accuracy ≈ 95% دون أن يكتشف أي تغيير. لذلك
نُبلغ عن:

| المقياس | التعريف | لماذا |
|---|---|---|
| Precision / Recall / F1 (change) | TP/(TP+FP) ، TP/(TP+FN) ، التوفيق بينهما | تركز على فئة التغيّر الأهم |
| IoU (change) | TP/(TP+FP+FN) | المقياس القياسي في التجزئة |
| **Mean IoU** | متوسط IoU للفئتين | يمنع اختفاء فئة |
| **Balanced Accuracy** | متوسط recall الفئتين | مقاوم لعدم التوازن |
| **Cohen's Kappa** | (p_o − p_e)/(1 − p_e) | يصحح الاتفاق العشوائي |
| Accuracy | (TP+TN)/total | تُعرض **للمعلومية فقط** |

**العتبة:** التنبؤ = `sigmoid(logits) > threshold`. يمكن ضبط العتبة على
**validation** (الخلية أدناه) بدل اعتبار 0.5.

المنطق الكامل غير التفاعلي: `python evaluate.py --threshold auto`."""),
    code(PY_PATH_SETUP + """
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src import config
from src.data_discovery import build_samples, download_dataset, load_mask
from src.dataset import EvalTileDataset, samples_for_split
from src.metrics import ConfusionMeter
from src.model import SiameseUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# اختار أفضل run (أعلى val_iou في checkpoint) وليس الأحدث — حتى لا يلتقط
# دفترُ التدريب التجريبي (run_nb_...) نفسَه بدل النموذج النهائي
runs = sorted(config.OUTPUT_DIR.glob("run_*"))
best_run, best_val = None, -1.0
for r in runs:
    ck_file = r / "best_model.pt"
    if not ck_file.exists():
        continue
    meta = torch.load(ck_file, map_location="cpu", weights_only=False)
    v = float(meta.get("val_iou", -1.0))
    print(f"  {r.name}: epoch {meta.get('epoch')} val_iou {v:.4f}")
    if v > best_val:
        best_run, best_val = r, v
run_dir = best_run
ckpt_path = run_dir / "best_model.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = SiameseUNet(in_ch=config.IN_CHANNELS, base=32).to(device)
model.load_state_dict(ckpt["model"]); model.eval()
print(f"\\nselected run: {run_dir.name} | checkpoint epoch {ckpt.get('epoch')} | "
      f"val IoU {ckpt.get('val_iou', float('nan')):.4f}")

root = download_dataset()
samples, _ = build_samples(root)
val_s, test_s = samples_for_split(samples, "val"), samples_for_split(samples, "test")
val_ds = EvalTileDataset(val_s, image_size=config.IMAGE_SIZE)
test_ds = EvalTileDataset(test_s, image_size=config.IMAGE_SIZE)
val_dl = DataLoader(val_ds, batch_size=32, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=32, num_workers=0)
print(f"val tiles: {len(val_ds)} | test tiles: {len(test_ds)}")
"""),
    md("## 1) ضبط العتبة على validation"),
    code("""
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.701, 0.05)]
meters = {t: ConfusionMeter() for t in THRESHOLDS}

with torch.no_grad():
    for batch in val_dl:
        a = batch["a"].to(device); b = batch["b"].to(device)
        logits = model(a, b).float().cpu()
        metas = batch["meta"]
        for j in range(logits.shape[0]):
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            lg = logits[j, :, :vh, :vw]; yt = batch["mask"][j, :, :vh, :vw]
            for t in THRESHOLDS:
                meters[t].update(lg.unsqueeze(0), yt.unsqueeze(0), t)

rows = [(t, meters[t].metrics()) for t in THRESHOLDS]
print(f"{'thr':>5} | {'F1':>7} | {'IoU':>7} | {'P':>7} | {'R':>7}")
for t, m in rows:
    print(f"{t:5.2f} | {m['f1']:7.4f} | {m['iou']:7.4f} | {m['precision']:7.4f} | {m['recall']:7.4f}")
THRESHOLD = max(rows, key=lambda x: x[1]["f1"])[0]
print(f"\\nchosen threshold (best val F1): {THRESHOLD}")
"""),
    md("## 2) التقييم النهائي على test (مرة واحدة)"),
    code("""
meter = ConfusionMeter()
with torch.no_grad():
    for batch in test_dl:
        a = batch["a"].to(device); b = batch["b"].to(device)
        logits = model(a, b).float().cpu()
        metas = batch["meta"]
        for j in range(logits.shape[0]):
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            meter.update(logits[j, :, :vh, :vw], batch["mask"][j, :, :vh, :vw], THRESHOLD)
m = meter.metrics()
total = m["tp"] + m["fp"] + m["fn"] + m["tn"]

print("== CHANGE-class (headline) ==")
print(f"Precision {m['precision']:.4f} | Recall {m['recall']:.4f} | "
      f"F1 {m['f1']:.4f} | IoU {m['iou']:.4f}")
print("\\n== Aggregate ==")
print(f"Mean IoU {m['mean_iou']:.4f} | Balanced Accuracy {m['balanced_accuracy']:.4f} | "
      f"Kappa {m['kappa']:.4f}")
print(f"Overall Accuracy {m['accuracy']:.4f}  <-- مضللة: "
      f"{(m['tn']+m['fp'])/total*100:.2f}% من البكسل غير متغيرة")
print("\\n== Per-class ==")
print(f"{'class':>10} | {'precision':>9} | {'recall':>9} | {'f1':>9} | {'iou':>9}")
print(f"{'change':>10} | {m['precision']:9.4f} | {m['recall']:9.4f} | {m['f1']:9.4f} | {m['iou']:9.4f}")
print(f"{'unchanged':>10} | {m['precision_unchanged']:9.4f} | {m['recall_unchanged']:9.4f} | "
      f"{m['f1_unchanged']:9.4f} | {m['iou_unchanged']:9.4f}")
"""),
    code("""
cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
cm_pct = cm / cm.sum() * 100
cm_row = cm / cm.sum(axis=1, keepdims=True) * 100
labels = [["TN", "FP"], ["FN", "TP"]]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, mat, title in ((axes[0], cm_pct, "% of all pixels"),
                       (axes[1], cm_row, "% of the TRUE class (row-normalized)")):
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\\n{cm[i,j]:,.0f}\\n{mat[i,j]:.1f}%",
                    ha="center", va="center",
                    color="white" if mat[i, j] > 50 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["pred: unchanged", "pred: change"])
    ax.set_yticks([0, 1], ["true: unchanged", "true: change"])
    ax.set_title(title, fontsize=10)
fig.suptitle(f"Pixel confusion matrix (test, threshold={THRESHOLD})")
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=110)
plt.show()
"""),
    md("## 3) أمثلة بصرية: A / B / ground truth / prediction"),
    code("""
N_VIS = 4
ids = list(range(min(N_VIS, len(test_s))))

canvas = {i: np.zeros((1024, 1024), dtype=np.uint8) for i in ids}
seen = {i: set() for i in ids}
with torch.no_grad():
    for batch in test_dl:
        a = batch["a"].to(device); b = batch["b"].to(device)
        logits = model(a, b).float().cpu()
        metas = batch["meta"]
        for j in range(logits.shape[0]):
            sid = int(metas["sample_idx"][j])
            if sid not in ids:
                continue
            pred = (torch.sigmoid(logits[j, 0]) > THRESHOLD).numpy().astype(np.uint8)
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            top, left = int(metas["top"][j]), int(metas["left"][j])
            canvas[sid][top:top+vh, left:left+vw] = pred[:vh, :vw]
            seen[sid].add((top, left))
        if all(len(seen[i]) == 16 for i in ids):   # 16 tile لصورة 1024
            break

fig, axes = plt.subplots(len(ids), 4, figsize=(17, 4.2 * len(ids)))
for row, i in enumerate(ids):
    s = test_s[i]
    axes[row, 0].imshow(np.array(Image.open(s.a_path).convert("RGB")))
    axes[row, 0].set_title("t1 (A)")
    axes[row, 1].imshow(np.array(Image.open(s.b_path).convert("RGB")))
    axes[row, 1].set_title("t2 (B)")
    gt = load_mask(s.mask_path)
    axes[row, 2].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title(f"ground truth ({gt.mean()*100:.1f}%)")
    axes[row, 3].imshow(canvas[i], cmap="gray", vmin=0, vmax=1)
    axes[row, 3].set_title(f"prediction (thr={THRESHOLD})")
    for c in range(4):
        axes[row, c].axis("off")
plt.tight_layout()
plt.show()
"""),
    code("""
# حفظ ملخص النتائج داخل مجلد الـrun
import json
out_dir = run_dir / f"eval_nb_{time.strftime('%Y%m%d_%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "test_metrics.json").write_text(
    json.dumps({"threshold": THRESHOLD, **m}, indent=2), encoding="utf-8")
print("saved:", out_dir / "test_metrics.json")
"""),
])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, obj in (("00_setup.ipynb", SETUP),
                      ("01_data_exploration.ipynb", EXPLORATION),
                      ("02_training.ipynb", TRAINING),
                      ("03_evaluation.ipynb", EVALUATION)):
        path = OUT / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        json.loads(path.read_text(encoding="utf-8"))  # validate
        print("wrote", path)


if __name__ == "__main__":
    main()
