# Cephalometric Landmark Detection

This project explores how artificial intelligence can help locate anatomical landmarks on lateral cephalometric X-rays. It was developed from a dental-student perspective: the goal is to connect familiar cephalometric anatomy with a practical computer-vision workflow.

The model predicts 19 landmark locations. The notebooks show the steps from training to fine-tuning and a small cephalometric angle analysis.

> **Important:** This is a research and learning project, not a diagnostic tool. Model predictions must not replace a clinician's assessment. Do not use the results for patient care without appropriate clinical validation.

## What is in this repository?

- [`Notebooks/`](Notebooks/) — guided, step-by-step experiments:
  - `01-pretraining.ipynb` — trains the initial landmark-detection model.
  - `02-finetuning-and-evaluation.ipynb` — adapts the model and evaluates landmark predictions.
  - `02-finetuning-and-evaluation_fa.ipynb` — Persian-language version of the fine-tuning notebook.
  - `03-cephalometric-analysis.ipynb` — explores SNA, SNB, and ANB measurements from predicted landmarks.
- [`src/`](src/) — the Python code used by the notebooks for data handling, the model, training, evaluation, and analysis.

## A simple way to follow the project

1. Start with the pre-training notebook.
2. Continue to the fine-tuning and evaluation notebook.
3. Explore the angle-analysis notebook after you understand how landmarks are predicted.

The notebooks may expect datasets and trained model checkpoints that are not included in this repository. Follow the setup notes inside each notebook and update its file paths for your own environment. Never upload identifiable patient images or private clinical data to a public repository.

## What do the measurements mean?

SNA, SNB, and ANB are angles commonly used in orthodontic assessment. In this project, they are calculated from predicted landmarks to demonstrate a research workflow. Landmark errors can affect the calculated angles, so results should be interpreted cautiously and checked against the source images and appropriate clinical references.

## Limitations

- Results depend on the data, image quality, landmark definitions, and model checkpoint used.
- Validation results are not the same as testing on a fully independent group of images.
- The notebooks may need adjustment to run outside the environment in which they were developed.
- This repository does not provide clinical recommendations or a validated medical device.

## Acknowledgement

This repository is shared for education and research into cephalometric landmark detection. Please review dataset terms and obtain the necessary permissions before using any data.
