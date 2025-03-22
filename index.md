---
layout: default
# <!-- title: "🚀 Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detection"
# <!-- author: "Anonymous Authors"
# <!-- description: "Zero-shot Anomaly Detection Model"
---

# 🚀 Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detection

<!-- **Anonymous Authors**  
_(Under Review)_ -->

[📄 Paper (Coming Soon)]() | [💾 Code (Coming Soon)]()

---

## 🔍 Abstract

Anomaly Detection (AD) is crucial for medical diagnostics and industrial defect detection. Traditional AD methods rely on normal training samples, but collecting such data is often impractical. Additionally, these methods struggle with generalization across domains.  

Recent advancements like **AnomalyCLIP** and **AdaCLIP** leverage CLIP’s zero-shot generalization but face challenges in bridging the gap between **image-level and pixel-level anomaly detection**.  

🚀 **Crane** improves upon these by:
- **Context-Guided Prompt Learning**: Dynamically conditioning text prompts using image context.  
- **Attention Refinement**: Modifying the CLIP vision encoder to enhance feature extraction for fine-grained anomaly detection.  

Our method **achieves state-of-the-art results**, improving image-level detection accuracy by **0.9\% to 4.9\%** and pixel-level anomaly localization by **2.8\% to 29.6\%** across **14 datasets**, demonstrating its effectiveness at both anomaly localization and detection.

---

## 📊 Quantitative Comparison with SOTA
Unlike AnomalyCLIP and AdaCLIP, **Crane** consistently improves both localization  and detection, setting new benchmark for zero-shot anomaly detection.

📌 **Image-level AP and pixel-level AUPRO measurements across 7 diverse industrial datasets:**
  
![Quantitative Results](assets/dual_radar.svg)

---

## 🖼️ Qualitative Comparison with SOTA
AdaCLIP and VAND struggle to maintain a balance between true positive and false negative rates. AnomalyCLIP further enhances sensitivity but continues to exhibit a high false negative rate, limiting its effectiveness. In contrast, Crane benefits from a stronger semantic correlation among patches, which improves the true positive rate while reducing false positives simultaneously.


📌 **Localization comparison for SOTA models, across various industrial textures and anomalous patterns:**

![Qualitative Results](assets/crane_qualitative_comparison.png)


📌 **Localization outputs of Crane, across various industrial textures and anomalous patterns:**

![Qualitative Results](assets/crane_qualitative_comparison.png)

<p style="text-align: center;">
  <input type="range" min="1" max="3" value="1" id="slider" oninput="updateImage()">
</p>

<p style="text-align: center;">
  <img id="imageDisplay" src="assets/crane_qualitative_comparison.png" style="max-width: 80%; height: auto;">
</p>

<script>
  function updateImage() {
    var slider = document.getElementById("slider");
    var image = document.getElementById("imageDisplay");
    var images = ["assets/crane_qualitative_comparison.png", "assets/image2.png", "assets/image3.png"];  // Add your images here
    image.src = images[slider.value - 1];
  }
</script>

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