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




### 🟧 ESPERIMENTO 06 — ResNet18 Full Fine-Tuning con Dropout (p = 0.3)

> [!IMPORTANT]
> **Obiettivo:** verificare se l'introduzione del dropout nel classificatore finale migliora la generalizzazione e riduce l'overfitting osservato negli esperimenti precedenti.

**Configurazione**
- Dataset: 10.000 immagini (5.000 FFHQ reali, 5.000 StyleGAN2 fake).
- Modello: ResNet18 pre-addestrata su ImageNet, full fine-tuning.
- Ottimizzatore: AdamW, weight decay = 0.01.
- Learning rate: \(10^{-5}\).
- Dropout: \(p=0.3\), applicato alle 512 feature prima del classificatore.
- Batch size: 32; epoche: 10.

**Risultati principali**

| Metrica | Valore |
|---|---:|
| Training accuracy finale | 99,71% |
| Validation accuracy finale | 77,93% |
| Migliore validation loss (epoca 5) | 0,5334 |
| Validation accuracy all'epoca 5 | 74,13% |
| Test accuracy (checkpoint migliore) | 74,00% |
| Test loss (checkpoint migliore) | 0,5287 |

**Osservazioni**

Il dropout ha rallentato leggermente l'apprendimento, ma non ha eliminato l'overfitting. La training accuracy raggiunge comunque quasi il 100%, mentre la validation loss torna a crescere dopo la quinta epoca, nonostante l'accuracy continui a migliorare.

Rispetto al precedente full fine-tuning senza dropout, la migliore validation loss è peggiorata da **0,5126 a 0,5334**, mentre la test accuracy del checkpoint selezionato è diminuita dal **75,80% al 74,00%**.

**Conclusione:** il dropout con \(p=0.3\), applicato soltanto al classificatore finale, non ha prodotto benefici evidenti nella configurazione sperimentata. Nei prossimi esperimenti sarà opportuno valutare altre strategie di regolarizzazione, in particolare data augmentation e learning rate scheduling.



### 🟧 EXP07 — Data augmentation (Horizontal Flip + Dropout)

**Configurazione:** ResNet18 con full fine-tuning, 15 epoche, learning rate \(10^{-5}\), weight decay \(0.01\), dropout \(p=0.3\) e `RandomHorizontalFlip(p=0.5)` applicato esclusivamente al training set.

**Risultati:** la migliore validation loss è **0.4971 all'epoca 8**, con validation accuracy del 78.13%. La massima validation accuracy raggiunge **80.47% all'epoca 14**. Il checkpoint migliore, selezionato sulla validation loss, ottiene **77.27% di test accuracy** e test loss pari a **0.4954**.

**Osservazioni:** rispetto agli esperimenti precedenti, la validation loss raggiunge un minimo inferiore e più tardivo. Tuttavia, dopo l'epoca 8 ricomincia a crescere mentre la training loss continua a diminuire, indicando che l'overfitting persiste. Le 15 epoche confermano che prolungare il training non migliora necessariamente la generalizzazione misurata con la Cross-Entropy.

**Nota:** verificare l'ottimizzatore effettivamente utilizzato, poiché il log riporta `Adam` anziché `AdamW`.

**W&B:** [EXP07 — Augmentation + Dropout](https://forge.coreweave.com/wandb/pasqem-university-of-florence/xai-counterfactuals/runs/pv7125zx)



### 🟧 EXP08 — AdamW + Augmentation + Dropout 0.08

**Configurazione:** ResNet18 full fine-tuning, AdamW (`lr=1e-5`, `weight_decay=0.01`), dropout `p=0.08`, `RandomHorizontalFlip(p=0.5)` e 15 epoche.

**Risultati:**
- Migliore validation loss: **0.4860 (epoca 8)**, con accuracy del 78.33%.
- Massima validation accuracy: **81.40% (epoca 14)**.
- Test accuracy del miglior checkpoint: **78.20%**, test loss 0.4837.
- Confusion matrix sulla validation: `[[597, 153], [172, 578]]`, con recall Fake 79.6% e Real 77.1%.

**Osservazioni:** leggero miglioramento rispetto a EXP07 (validation loss 0.4971, test accuracy 77.27%), ma il confronto non isola l'effetto del dropout perché è cambiato anche l'ottimizzatore. L'overfitting persiste: dopo l'epoca 8 la validation loss ricomincia a crescere, mentre la training accuracy raggiunge il 99.76%.

**Prossimo esperimento:** introdurre `ReduceLROnPlateau` per verificare se la riduzione adattiva del learning rate permette di migliorare la generalizzazione.

**W&B:** [EXP08 — AdamW + Augmentation + Dropout 0.08](https://forge.coreweave.com/wandb/pasqem-university-of-florence/xai-counterfactuals/runs/fvwwsjnv)




### 🟧 EXP09 — Learning Rate Scheduler + Early Stopping

**Configurazione:** ResNet18 full fine-tuning, AdamW (`lr=1e-5`, `weight_decay=0.01`), dropout `p=0.08`, `RandomHorizontalFlip(p=0.5)`, massimo 25 epoche. Introdotti `ReduceLROnPlateau` (`factor=0.5`, `patience=2`) ed early stopping (`patience=8`).

**Risultati:** migliore validation loss **0.4860 all'epoca 8**, identica a EXP08. Il learning rate viene dimezzato alle epoche 11 e 14, senza ottenere nuovi minimi. L'early stopping interrompe il training all'epoca 16. Il miglior checkpoint raggiunge **78.20% di test accuracy**, con test loss **0.4837**.

**Osservazioni:** lo scheduler rallenta il peggioramento della validation loss nelle ultime epoche, ma non migliora il checkpoint ottimale. L'early stopping funziona correttamente, evitando 9 epoche aggiuntive. L'overfitting persiste.

**Conclusione:** mantenere scheduler ed early stopping per i futuri esperimenti, concentrandosi ora su nuove strategie di data augmentation.

**W&B:** [EXP09 — Scheduler + Early Stopping](https://forge.coreweave.com/wandb/pasqem-university-of-florence/xai-counterfactuals/runs/nxigpj0z)




### 🟧 EXP10 — ResNet50 Full Fine-Tuning

**Configurazione:** sostituita ResNet18 con ResNet50 preaddestrata su ImageNet (23,5M parametri addestrabili). AdamW (`lr=1e-5`, `weight_decay=0.01`), dropout `p=0.08`, `RandomHorizontalFlip(p=0.5)`, scheduler `ReduceLROnPlateau` ed early stopping (`patience=8`). Massimo 25 epoche.

**Risultati:** migliore validation loss **0.2029 all'epoca 14**, con validation accuracy **91.87%**. Massima validation accuracy **92.87%** (epoca 20). Il miglior checkpoint raggiunge **92.47% di test accuracy** e test loss **0.2175**.

**Osservazioni:** miglioramento netto rispetto a ResNet18 (78.20% di test accuracy). La validation loss rimane relativamente stabile dopo il minimo, nonostante la training accuracy raggiunga quasi il 100%. Lo scheduler riduce il learning rate alle epoche 17 e 20 senza migliorare ulteriormente il checkpoint. Early stopping all'epoca 22.

**Conclusione:** ResNet50 rappresenta un miglioramento significativo e viene scelta come classificatore di riferimento per i successivi esperimenti sui controfattuali.

**W&B:** [EXP10 — ResNet50](https://forge.coreweave.com/wandb/pasqem-university-of-florence/xai-counterfactuals/runs/pm04gtzz)





## Week 4 - November 23-29, 2026
### Configurazione ambiente Alibi

- **Creazione di `XAIalibi`:** ambiente virtuale separato da `XAIvenv`, per evitare conflitti tra le dipendenze di Alibi e quelle utilizzate per addestrare ResNet50.
- **Python 3.11:** installato perché più compatibile con Alibi 0.9.6 e con le librerie legacy necessarie ad alcuni metodi controfattuali.
- **Aggiornamento di `pyenv`:** necessario perché la versione iniziale non riconosceva Python 3.11.18.
- **Problema di compilazione:** l'installazione di Python falliva per un'incompatibilità tra il linker di macOS e l'SDK 27.0.
- **Soluzione:** installazione di Python tramite `SDKROOT`, utilizzando temporaneamente l'SDK 26.5 già presente sul Mac, senza modificare la configurazione globale.
- **Installazione di Alibi 0.9.6:** effettuata nel nuovo ambiente `XAIalibi`, mantenendo intatto `XAIvenv`.
- **Errore di importazione:** Alibi non riusciva a caricare spaCy per l'assenza della dipendenza `click`. Risolto installando `click>=8,<9`.
- **Verifica finale:** `pip check` senza conflitti dichiarati e importazione di Alibi riuscita.

**Risultato:** ambiente `XAIalibi` funzionante per iniziare gli esperimenti con Alibi. Restano da installare e verificare le dipendenze specifiche dei metodi controfattuali scelti, incluso PyTorch per utilizzare ResNet50.