# Global Image Geolocation

**2nd place** — Geolocation Challenge, AI Club Hackathon Vertical, IIT Madras (August 2026)

Given a single photograph, predict where on Earth it was taken: latitude, longitude, and a
confidence radius in kilometres. No external APIs, no reverse image search, no geocoding
services at inference time. Free-tier GPUs only.

---

## The problem, and what it actually rewards

Each prediction is scored on three things: how close the coordinate is (with a smooth
distance decay), how well the claimed radius matches the actual error, and whether the
country is right. The final score is the **median** across all test images, not the mean.

Two consequences drove almost every design decision:

**Median scoring means outliers are free.** Being catastrophically wrong on 10% of images
costs nothing. There is no reward for heroics on hard images. You win by lifting the
typical image.

**Errors at this scale are large.** The OSV-5M paper's best model, trained on 4.9M images,
averages 1,814 km of error. Individual human annotators average 6,407 km. So the distance
term is small for most images no matter what you do, and the **country bonus and radius
calibration are where the recoverable points are**. We spent a lot of early effort chasing
coordinate precision before working this out.

---

## Approach

We treat this as classification over geographic cells, not coordinate regression.

Direct latitude/longitude regression fails in a specific way: if a photo could plausibly
be Ireland or New Zealand, a model minimising coordinate error outputs something between
them, which is the middle of the Atlantic. Classification lets the model say "60% here,
30% there" instead of averaging two answers that should never be averaged.

```
                     photo
                       |
          +------------+------------+
          |                         |
   CLIP ViT-L/14            DINOv2-large        (both frozen, never trained)
   image-text trained    self-supervised
          |                         |
          +------------+------------+
                       |
              4,096 features per image
                       |
                 trained head (MLP)
                       |
    +---------+--------+--------+---------+
    |         |        |        |         |
  fine     coarse   country   offset   uncertainty
 (2,000)    (150)    (298)     (3)        (1)
    |         |        |        |         |
    +---------+--------+--------+---------+
                       |
           latitude, longitude, radius
```

### Why two frozen backbones

CLIP was trained to match images with captions. DINOv2 was trained self-supervised with no
labels at all. Different objectives mean they are wrong about different images, so
combining them is worth more than combining two similar models. Adding DINOv2 with nothing
else changed took us from 78 to 82.

Neither was trained to predict location, which is what made them permissible: the
organisers banned any model trained on the geolocation objective, including StreetCLIP and
GeoCLIP, and including their embeddings or distillation from them.

Because the backbones are frozen, we ran every image through them once and cached the
feature vectors. That drops head training from hours to about four minutes, which is the
only reason a 5-model ensemble and a full radius search were affordable on free-tier GPUs.

### Distance-aware soft targets

Plain cross-entropy treats every wrong cell as equally wrong. Guessing the next city over
and guessing another continent both just count as "wrong", which is nonsense for geography.

Instead of a one-hot target, we build a target spread across all cells weighted by
`exp(-d / τ)`, where `d` is the real great-circle distance between cell centres and
`τ = 250 km`. Near misses get most of the credit. It also means thinly-populated cells
still learn something from every nearby image.

### The radius is a learned output, not a heuristic

The radius is scored directly, so we modelled it. A dedicated head is trained to predict
`log(1 + error)` using a **pinball (quantile) loss at q = 0.70**. Quantile regression fits
because the scoring is asymmetric: we want a radius that *usually* contains the truth, not
one that matches the average error.

---

## Three bugs worth reading about

These were the most instructive parts of the project.

### 1. A quarter of our predictions were in the ocean

Our first decoder averaged the top-K cell centroids in 3D, weighted by probability. This is
wrong in a way the loss will never surface. If the top cells are three in France plus Sydney
and Tokyo, their average lands in open water. In one run **24% of predictions fell outside
every country polygon**.

We only found it by looking at the actual output coordinates. Reproduced on a constructed
case: a France-leaning prediction with moderate spread decoded to **latitude 57, longitude
21 — the middle of the Baltic Sea**.

Fix: only average cells within 600 km of the top-ranked cell, and drop the rest. The top
cell is at distance zero from itself, so the weights can never all cancel. Same constructed
case now returns Paris. Ocean predictions dropped to 87 of 2,448 (3.6%).

### 2. Our validation set was lying to us

Street-view images come in sequences: the same vehicle photographing the same road seconds
apart. A random train/validation split puts near-identical frames on both sides, so the
model recognises them rather than geolocating them. Validation looked great, the leaderboard
disagreed.

We split by **spatial blocks** instead, assigning whole geographic clusters to one side.
Block size turned out to matter more than expected:

| Blocks | Reported median error | What it was actually measuring |
|---|---|---|
| 60 (large) | 1,460 km | extrapolation to unseen continents |
| 1,583 (small) | 583 km | the actual task |

**The model did not change between those two numbers. Only the measurement did.**

### 3. Filename matching cannot detect test contamination

The rules forbid training on hidden test images even accidentally. But the competition
renamed all its images, so a competition `image_id` and an OSV-5M id can never collide by
construction. A filename check returns zero overlap whether or not the same photo is in
both sets. It looks like a pass and means nothing.

We compared **image content** with a DCT perceptual hash (64-bit), which survives resizing
and re-compression, and dropped anything within a Hamming distance of 8 of a test image. We
validated that threshold empirically: a resized and re-compressed copy scored 6, unrelated
images scored around 32.

It removed **266 images**. OSV-5M genuinely does overlap the competition data.

---

## Results

Held-out set: 3,908 provided images, excluded from every model in the ensemble.

| Statistic | Value |
|---|---|
| Median error | 583 km |
| Mean error | 2,408 km |
| p10 / p25 / p75 / p90 | 138 / 271 / 2,121 / 8,762 km |
| Country head accuracy | 64.0% |
| Predicted point in correct country | 50.4% |
| Radius coverage | 58.4% |

For scale: the OSV-5M paper's best model, trained on **4.9M images**, reports 1,814 km mean
error and 68% country accuracy. We used roughly 5% of that data.

### Calibration: the errors come in two groups, not one

The error histogram is clearly bimodal. Splitting at 3,000 km:

| Group | Share | Median error | Radius coverage |
|---|---|---|---|
| Roughly right region | 79.6% | 428 km | **73.4%** |
| Wrong continent | 20.4% | all > 3,000 km | **0.0%** |

That second group explains the 58.4% headline. Those are not near misses. The model picked
the wrong continent, and since the radius is capped at 3,000 km, **no radius we are allowed
to emit could ever cover them**. On the 79.6% where the region is roughly right, coverage
is 73.4%, close to the 70% quantile the head was trained to target. The calibration works
on the images where calibration is possible at all.

### Measuring an undisclosed scoring function

The exact scoring formula was never published, only its shape. Rather than guess one
formula and tune hard against it, we picked the radius policy with the best **worst case**
across a family of plausible scoring functions.

Then we realised we could measure it directly. We submitted the **same coordinates** three
times, varying only the radius column:

| Radius scale | Score |
|---|---|
| 0.5x | 70 |
| 1.0x | **82** |
| 2.0x | 79 |

Since the coordinates never change, every point of difference comes from the calibration
term alone. This confirmed our operating point was near optimal, and that being too tight
hurts far more than being too wide (halving cost 12 points, doubling cost 3).

It also agrees with the coverage analysis: widening to 2x only lifts coverage from 58% to
about 73%, because the wrong-continent group stays uncovered regardless.

**Known weakness:** the rank correlation between claimed radius and actual error is only
**0.30**. The head can tell "this one is hard" but not "this one is hopeless", and the
hopeless cases are exactly that 20.4%. Teaching the model to flag its own wrong-continent
failures is the first thing we would do with more time.

---

## What did not work

| Tried | Outcome |
|---|---|
| Direct lat/lon regression | Averaged contradictory cues into the ocean. Abandoned. |
| One-hot cross-entropy | Punishes a neighbouring cell as hard as the wrong continent. Dropped for soft targets. |
| Entropy-based radius | Entropy is distance-blind: two nearby towns look identical to two continents. |
| 400 → 3,000 geocells | Score moved 1 point. Within noise. Told us the errors were about picking the wrong *region*, not imprecision within the right one. |
| Backbone fine-tuning | Inconclusive. Hit a wall-clock limit at step 2000/4687 of epoch 3, so the LR schedule never annealed and loss was still falling. Undertrained, not disproven. |
| Scaling to 519K images | Score dropped 82 → 69. Diagnosed from the output, not the metrics: *every* radius came back pinned at the 3,000 km cap, the signature of features misaligned with labels. |
| Isotonic radius calibration | Expected it to beat a simple multiplier. It was 4.6% *worse* on synthetic data. We shipped both and let held-out data choose. |

The cell-count result was the most useful negative finding. It redirected our effort from
geographic resolution to the country head and radius calibration, which is where the
remaining points actually were.

---

## Repository layout

```
notebooks/
  approach2-part1-geogs-v6.ipynb         CLIP-only training run (78)
  approach2-part2-final-v6.ipynb         v6 inference on final test set
  approach2-part3-geogs-v7.ipynb         CLIP + DINOv2 training run (82)
  approach2-part4-final-v7.ipynb         v7 inference, produced the submission
  approach2-part5-validation.ipynb       held-out validation + calibration figures
```

## Reproducing

**Environment.** Kaggle Notebook, Python 3.11+, single T4 (free tier).
`transformers==4.44.2`, `huggingface_hub`, `shapely`, `scikit-learn`, `pandas`, `numpy`,
`pillow`, `scipy`, `openpyxl`. Everything else ships in the default Kaggle image.

**Order.** Run the v7 training notebook first. It writes all weights and calibration
parameters to `/kaggle/working/artifacts`. Attach that notebook's output to the inference
notebook and run it.

**Runtime.** Training is about 3 hours on one free-tier T4, roughly two thirds of which is
feature extraction. Inference on 2,448 images takes under 3 minutes.

**Internet.** On for training, since external data is pulled from Hugging Face. **Not
needed for inference** — that is the point below.

### Offline inference

All backbone weights are saved to disk during training and reloaded from local paths at
inference. The inference notebook sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
before importing Transformers, and includes a cell that reloads every weight with the hub
disabled and prints pass/fail. It proves the claim rather than making it.

### Reproducibility

All seeds fixed (Python, NumPy, PyTorch, CUDA), `cudnn.deterministic` enabled, and every
file listing sorted before sampling. That last one sounds pedantic but matters: `glob`
returns files in filesystem order, so without sorting, the external data sample would not
reproduce on a different machine even with a fixed seed. It caught us once.

---

## Data

- **Provided:** 19,002 geotagged images.
- **External:** 250,000 images from [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m),
  sampled from the first five training shards.
- **After contamination filtering:** 268,736 images.

We did not use `osv5m/baseline`, the model published alongside that dataset, since it is
trained for geolocation and was explicitly disallowed.

## References

- [OpenStreetView-5M: The Many Roads to Global Visual Geolocation](https://arxiv.org/abs/2404.18873) — benchmark numbers we used to calibrate expectations
- [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) — frozen backbone
- [DINOv2-large](https://huggingface.co/facebook/dinov2-large) — second frozen backbone
- Natural Earth Admin 0 boundaries — country labels, constrained decoding, land snapping

