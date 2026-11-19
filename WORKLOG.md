# XAI Counterfactuals — Work Log

This document tracks the weekly progress of the Explainable AI project, including completed tasks, technical decisions, encountered issues, and planned activities.

---

## Week 1 — October 27 – November 1, 2026

### Project setup

- Created the GitHub repository `xai-counterfactuals`.
- Defined the initial project structure, separating source code, counterfactual explainers, experiments, datasets, checkpoints, results, and slides.
- Added `main.py`, `requirements.txt`, and `.gitignore`.
- Reviewed the structure and coding style of the previous `autoformer-from-scratch` project as a reference.
- Started the central configuration file `src/config.py`.

### Local environment

- Created the virtual environment `XAIvenv` using pyenv and Python 3.12.4.
- Installed PyTorch 2.14.1 and Torchvision 0.29.1.
- Verified that Apple Metal Performance Shaders (MPS) is available.
- Installed the initial project dependencies from `requirements.txt`.
- Fixed a missing OpenSSL dependency affecting Python's SSL support.

### Remote computing resources

- Verified SSH access to the University of Florence server.
- Identified two NVIDIA RTX A2000 GPUs with 12 GB VRAM each.
- Created a separate Python virtual environment on the server.
- Cloned the project repository on the server.
- Decided to develop and test locally first, using the server for computationally demanding experiments.

### Methodological decisions

- Start with a pretrained ResNet18 in PyTorch.
- Use FFHQ as the source of real face images, complemented by generated face images.
- Investigate all four counterfactual methods available in Alibi.
- Consider extending the experiments to a Vision Transformer if the initial results are promising.

### Next steps

- [ ] Finalize the project configuration.
- [ ] Investigate and prepare the real/fake face dataset.
- [ ] Implement image preprocessing and data loading.
- [ ] Load and adapt ResNet18 for binary classification.

### Notes




---

## Week 2 — November 16 – November 22, 2026

### Dataset preparation — November 16, 2026

- Downloaded 1,000 real FFHQ images at 1024×1024 resolution.
- Stored the images in `data/real/`.
- Verified image loading using Pillow.
- Confirmed RGB format and expected image dimensions.
- Successfully visualized a sample image in the exploration notebook.
- Next step: obtain generated face images for the fake class.




## Week 3 — November 16–22, 2026

### November 16 — Dataset preparation and verification

**Real images — FFHQ**
- Downloaded 1,000 real face images from the official NVIDIA FFHQ dataset.
- Stored the images in `data/real/`.
- Verified the image count, RGB format and original resolution (1024 × 1024).

**Fake images — StyleGAN2**
- Identified the official NVIDIA StyleGAN2 generated-image dataset.
- Selected the FFHQ 1024 × 1024 images generated using truncation parameter \(\psi=1.0\).
- Downloaded 1,000 generated faces from the `000000.zip` archive.
- Extracted the images into `data/fake/`.

**Dataset pipeline — `src/dataset.py`**
- Confirmed the class mapping: `fake = 0`, `real = 1`.
- Applied resizing to 224 × 224, tensor conversion and ImageNet normalization.
- Verified the reproducible 70% / 15% / 15% random split.

### November 17 — Model implementation

**ResNet18 — `src/models.py`**

- Implemented the `ResNet18Classifier` class, inheriting from `torch.nn.Module`.
- Loaded ResNet18 with pretrained ImageNet-1K weights using Torchvision.
- Replaced the original 1,000-class fully connected layer with a binary classifier (`512 → 2`).

**Training pipeline — `src/train.py`**

- Implemented a feature-extraction baseline with frozen pretrained layers.
- Kept the final fully connected layer trainable.

| Model parameters | Count |
|---|---:|
| Total | 11,177,538 |
| Frozen | 11,176,512 |
| Trainable | 1,026 |

---

### 🟧 EXPERIMENT 01 — ResNet18 feature extraction, LR = 1e-4

> [!IMPORTANT]
> **First baseline experiment.** ResNet18 pretrained on ImageNet, with all pretrained parameters frozen and only the final linear classifier optimized.

**Configuration**

| Hyperparameter | Value |
|---|---|
| Architecture | ResNet18 |
| Pretrained weights | ImageNet-1K |
| Training strategy | Feature extraction |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | 10 |
| Batch size | 32 |
| Loss | Cross-Entropy |

**Results — Epoch 10**

| Metric | Value |
|---|---:|
| Training loss | 0.6909 |
| Training accuracy | 53.29% |
| Validation loss | 0.6901 |
| Validation accuracy | 54.00% |

**Checkpoint:** `checkpoints/resnet18_best.pth`

### 🟧 EXPERIMENT 02 — ResNet18 feature extraction, LR = 1e-3

Identical to Experiment 01, except:

- Learning rate: `0.001`
- Separate checkpoint to preserve the first experiment.

**Results — Epoch 10**

| Metric | Value |
|---|---:|
| Training loss | 0.6204 |
| Training accuracy | 65.36% |
| Validation loss | 0.6886 |
| Validation accuracy reported during training | 57.00% |
| Validation accuracy from notebook confusion matrix | 56.00% |

**Checkpoint:** `checkpoints/resnet18_lr1e-3_best.pth`

**Preliminary conclusion**

Increasing the learning rate improves training-set performance but produces only a modest improvement in validation performance. This suggests that optimizing only the final linear classifier may be insufficient for the real/fake classification task, although further investigation is needed before drawing a definitive conclusion.


### 🟧 EXPERIMENT 03 — ResNet18 feature extraction, LR = 1e-3, dataset 10k, stratification

probabilmente il problema principale è la strategia di addestramento:

- In pratica, stiamo chiedendo a un classificatore lineare di riconoscere immagini generate da StyleGAN2 utilizzando 512 caratteristiche che ResNet18 ha imparato per riconoscere gli oggetti di ImageNet.
- Ma distinguere un volto reale da un volto generato è un problema particolare. Le due immagini possono condividere quasi tutte le caratteristiche semantiche importanti: occhi, naso, bocca, capelli, forma del viso.
- Le differenze potrebbero trovarsi in caratteristiche visive più sottili, come texture e artefatti di generazione, che l'estrattore congelato non rappresenta necessariamente in modo discriminante.
- Quindi aumentare il numero di immagini non cambia le caratteristiche che la rete è in grado di estrarre. Può aiutarci a stimare meglio la relazione tra le caratteristiche disponibili e la classe, ma non consente di apprendere nuove rappresentazioni.



### 🟧 EXPERIMENT 04 — ResNet18 Full Fine-Tuning (10k Images)

> [!IMPORTANT]
> **Objective:** Improve real/fake classification by fine-tuning all pretrained ResNet18 parameters instead of training only the final classification layer.

**Configuration**
- Dataset: 10,000 images (5,000 FFHQ real / 5,000 StyleGAN2 fake).
- Split: 7,000 training / 1,500 validation / 1,500 test (stratified).
- Initialization: ImageNet-1K pretrained weights, randomly initialized binary classification head.
- Training strategy: Full fine-tuning (11,177,538 trainable parameters).
- Optimizer: Adam, learning rate `1e-5`.
- Loss: Cross-Entropy.
- Batch size: 32; epochs: 10.
- Device: Apple MPS.
- Experiment tracking: Weights & Biases.

**Results**

| Metric | Value |
|---|---:|
| Final training accuracy | 99.91% |
| Final validation accuracy | 78.60% |
| Best validation loss (epoch 5) | 0.5126 |
| Validation accuracy at epoch 5 | 75.67% |
| Test accuracy (best checkpoint) | 75.80% |
| Test loss (best checkpoint) | 0.5106 |
| Average epoch duration | ~187 seconds |

**Observations**

- Full fine-tuning substantially improved classification performance compared with frozen-feature extraction.
- Training accuracy approached 100%, indicating strong adaptation to the training data.
- Validation loss started increasing after epoch 5, suggesting overfitting and increasing confidence in incorrect predictions.
- Validation accuracy continued improving until epoch 10, despite the increasing loss.
- The best checkpoint was selected at epoch 5 based on minimum validation loss.

**Conclusion and next steps**

Full fine-tuning significantly improved performance, but generalization remains limited. Future experiments will investigate regularization techniques, including dropout, weight decay, learning rate scheduling, and data augmentation.

**W&B run:** [resnet18full_lr1e-5](https://forge.coreweave.com/wandb/pasqem-university-of-florence/xai-counterfactuals/runs/7omnpppz)