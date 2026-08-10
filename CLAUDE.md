# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for **CITS5017 Deep Learning** (UWA, Semester 2 2026). It holds lab work and
assignments for the unit, plus the reference material they're based on. Most of it is
still empty: the lab sheets and the project spec are here, the notebooks answering them
get written as the semester goes.

Everything is TensorFlow/Keras. Unit coordinator is Du Huynh; the unit follows Aurélien
Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed).

## Environment

Always use the conda env, for every command that runs Python:

```bash
conda activate tf-2.20-gpu
```

It's defined by `tf-2.20-gpu.yml` (Python 3.12, TF 2.20 with CUDA, Keras 3.15,
keras-tuner, scikit-learn, transformers, gymnasium, jupyter). Rebuild with
`conda env create --file tf-2.20-gpu.yml` if it ever goes missing.

A GPU is visible to TF here. It's an sm_120 card and the TF 2.20 wheel has no prebuilt
kernels for compute capability 12.0, so the first run of a new kernel JIT-compiles from
PTX and can stall for a long while. That's expected, not a hang. The lab sheets also
mention UWA's HPC (Kaya) as a GPU option; nothing in this repo is set up for it.

Non-obvious version notes for code written against the textbook:
- Keras 3: `input_shape=` on the first layer is deprecated. Use
  `layers.InputLayer(shape=[...])` as the first layer instead.
- Models save as `.keras`, not `.h5`.

## Layout

- `lab1/`, `lab2/`, `lab3/` — one directory per weekly lab, each holding the lab sheet
  PDF and any data it needs (`lab1/circle.npy` is the 2D binary-classification set for
  lab 1 task 2). Labs are not assessed. Put new lab notebooks in the matching directory.
- `ass1/` — Project 1 (20%, due 11 Sep 2026): `project1.pdf` spec plus
  `train-2026.pkl` / `valid-2026.pkl` / `test-2026.pkl`.
- `handson-ml3/` — **nested git repo**, a clone of `ageron/handson-ml3`. Read-only
  reference: the textbook's official notebooks (`10_neural_nets_with_keras.ipynb`,
  `14_deep_computer_vision_with_cnns.ipynb`, etc). Don't commit changes into it, and
  don't stage it from the parent repo.
- `textbook_md/` — the textbook as markdown, see below.

## textbook_md

The same book parsed twice, `CH01`–`CH19` plus front/back matter, one file per chapter
from each parser:

- `*_marker.md` — cleaner prose, proper heading levels, no page furniture. Default to
  this one for reading. It references figure images (`![](_page_3_Figure_4.jpeg)`) that
  were **not** extracted, so those links are dead.
- `*_pymupdf.md` — keeps page numbers and a ProQuest watermark line on every page, and
  mangles hyphenation (`architec‐ ture`). Useful as a cross-check when marker drops or
  garbles something.

Chapter-to-topic mapping that matters here: 10 = ANNs/Keras, 11 = training deep nets,
14 = CNNs.

## Assignment 1 data

Pickles are dicts with **bytes keys**:

```python
def load_data(pickle_file):
    import pickle
    with open(pickle_file, "rb") as f:
        d = pickle.load(f)
    return d[b"data"], d[b"labels"], d[b"class_names"]
```

Shapes: train `(3700, 64, 64, 3)`, valid `(800, ...)`, test `(500, ...)`, all `uint8` in
0–255. 10 perfectly balanced classes: alligator, dragonfly, dugong, goldfish, goose,
koala, scorpion, slug, snail, tarantula. Instances are stored in increasing class-ID
order, so shuffle the training set before batching.

The spec (`ass1/project1.pdf`) pins down things worth not re-deriving: 4 conv layers +
4 pooling + dropout + global average pooling, under 30,000 trainable params;
`BayesianOptimization` tuner from keras-tuner with `max_trials=15` and 120 epochs per
trial; a `hyperparameter_tuning` boolean at the top of the notebook that switches
between tuning and loading the saved model; then fine-tuning `NASNetMobile` with a
≤80-neuron dense layer; then a comparison section. Deliverables are
`Surname_FirstName-proj1.ipynb`, `Surname_FirstName-CNN.keras`,
`Surname_FirstName-NASNetMobile.keras`, flat in one zip. Read the PDF before working on
it, this summary is not a substitute.

## Git

`origin` is `git@github.com:LeweiXu/CITS5017.git` (empty at time of writing, default
branch `master`). Nothing is committed yet. Before the first commit, sort out what
should actually be tracked: `handson-ml3/` is a foreign repo (74M), `ass1/*.pkl` is 59M,
and trained `.keras` files will be large too.
