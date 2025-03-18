---
layout: default
title: "Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detection"
---

# 🚀 Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detection

<!-- **Anonymous Authors**  
_(Under Review)_ -->

[📄 Paper (Coming Soon)]() | [💾 Code (Coming Soon)]()

---

## 🔍 Overview

Anomaly Detection (AD) is crucial for medical diagnostics and industrial defect detection. Traditional AD methods rely on normal training samples, but collecting such data is often impractical. Additionally, these methods struggle with generalization across domains.  

Recent advancements like **AnomalyCLIP** and **AdaCLIP** leverage CLIP’s zero-shot generalization but face challenges in bridging the gap between **image-level and pixel-level anomaly detection**.  

🚀 **Crane** improves upon these by:
- **Context-Guided Prompt Learning**: Dynamically conditioning text prompts using image context.  
- **Attention Refinement**: Modifying the CLIP vision encoder to enhance feature extraction for fine-grained anomaly detection.  

Our method **achieves state-of-the-art results**, improving accuracy by **2% to 10%** across **14 datasets**, demonstrating its effectiveness at both **image and pixel levels**.
---

## 📊 Quantitative Comparison with SOTA
We compare **Crane** with **AdaCLIP, AnomalyCLIP, and other SOTA methods** on multiple benchmark datasets.

📌 **Results Summary (Pixel-AUPRO & Image-AP on Various Datasets):**
  
![Quantitative Results](assets/dual_radar.pdf)

---

## 🖼️ Qualitative Comparison with SOTA
Below is a qualitative comparison between **Crane** and other methods, showcasing its superior anomaly localization.

📌 **Example Results on Various Datasets:**

![Qualitative Results](assets/crane_qualitative_comparison.pdf)

---

<!-- ## 🔬 Citation
If you find our work useful, please consider citing it:

```bibtex
@article{crane2024,
  title={Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detection},
  author={Anonymous Authors},
  journal={Under Review},
  year={2024}
} -->