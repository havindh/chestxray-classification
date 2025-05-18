
# Chest X-ray Classification

This CS231n project compares pretrained CNNs (DenseNet121) and a diffusion-based model (RoentGen) for multi-label chest X-ray classification. We evaluate both zero-shot and fine-tuned models on the NIH ChestX-ray8 dataset.

## ✅ Goals
- Run zero-shot inference using pretrained DenseNet121 from TorchXRayVision
- Fine-tune DenseNet121 on a subset of NIH images
- Evaluate both models on 4 target diseases: Cardiomegaly, Atelectasis, Pleural Effusion, Pneumothorax
- Compute per-label AUC and compare performance

## 📁 Structure
- `notebooks/` – Colab notebooks for zero-shot and fine-tuning
- `models/` – model loading & setup code
- `data/` – metadata, label splits (no raw images)
- `evaluation/` – AUC computation, plots
- `results/` – output AUCs, logs, plots
- `utils/` – helper functions (transforms, etc.)

## 🧪 Dataset
We use a subset of NIH ChestX-ray8 for evaluation and fine-tuning. Images are stored in Google Drive and not committed to this repo.

## 🔗 Dependencies
Install required packages:
```bash
pip install torch torchvision torchxrayvision scikit-learn matplotlib pandas tqdm
