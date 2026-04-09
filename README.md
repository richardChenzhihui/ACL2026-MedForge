# MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning

This repository contains the anonymized release of **MedForge-Reasoner**, including core training scripts, evaluation tools, and sample dataset components from the paper "MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning".

Arxiv Paper Link: https://arxiv.org/abs/2603.18577

## Abstract
Text-guided image editors can now manipulate authentic medical scans with high fidelity, threatening clinical trust and safety. MedForge provides a pre-hoc, evidence-grounded solution for medical forgery detection. We introduce **MedForge-90K**, a large-scale benchmark of realistic lesion edits with expert-guided reasoning supervision. **MedForge-Reasoner** performs localize-then-analyze reasoning, predicting suspicious regions before producing a verdict, and is aligned via Forgery-aware GSPO to strengthen grounding and reduce hallucinations.

## Project Structure
- `medforge_reasoner/`: Core logic for training (SFT/GSPO) and inference. Includes the Forgery-aware reward plugin.
- `data/`: Sample medical images (Real and Forgery) and their corresponding annotations.
- `evaluation/`: Scripts for LLM-as-judge evaluation and benchmark prompt generation.
- `dataset_tools/`: Tools used for generating high-fidelity forgeries and expert-guided annotations.
- `ms-swift/`: (To be installed) ModelScope SWIFT framework for efficient fine-tuning.

## Environment Setup
We recommend using a Conda environment with Python 3.10+.

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run setup script**:
   This script will clone `ms-swift` from GitHub and install it in editable mode, along with `flash-attention`.
   ```bash
   bash setup_env.sh
   ```

## Usage

### Supervised Fine-Tuning (SFT)
To start the SFT training:
```bash
cd medforge_reasoner
bash train.sh
```

### Training (GSPO)
To start the Forgery-aware GSPO training:
```bash
cd medforge_reasoner
bash train_gspo.sh
```

### Evaluation
The evaluation is divided into two primary stages: **Detection** and **Explainability**.

1. **Detection Evaluation (Core Metrics)**:
   This stage evaluates the model's ability to classify images as real or forgery and its localization accuracy. It computes classification Accuracy, F1 Score, and Mean Intersection over Union (mIoU) for bounding boxes.
   ```bash
   cd evaluation/detection_eval
   python eval.py --input /path/to/inference_results.jsonl --output ./results
   ```

2. **Explainability Evaluation (Reasoning Quality)**:
   This stage uses an LLM-as-judge approach to evaluate the quality and grounding of the model's generated reasoning (Chain-of-Thought).
   ```bash
   cd evaluation/explainability_llm_as_judge/gemini3pro-judge
   python judge_main.py
   ```

## Dataset
MedForge-90K contains:
- **30K Real Images** (CXR, Brain MRI, Fundus)
- **30K Lesion Implant Forgeries**
- **30K Lesion Removal Forgeries**

**Note**: The `data/` directory in this repository currently contains only sample data. Each forgery setting includes 10 samples for demonstration purposes. The full MedForge-90K dataset will be made available separately.

### Dataset CoT Example
The following is an example of the structured Chain-of-Thought (CoT) used for expert-guided reasoning:

```xml
<think>
Okay, let's see. I am starting to analyze whether this image is a deepfake: 
<description>This is an axial T2-weighted MRI of a brain</description>. 
I have identified the suspect deepfake area at: <|object_ref_start|>"deepfake"<|object_ref_end|><|box_start|>x1="350" y1="452" x2="596" y2="761"<|box_end|>. 
Now I am going to perform a detailed analysis of this suspect region. 
In my observation, the evidence for deepfake is: 
<evidence>The deepfake region displays a significant disruption of normal brain anatomy. The white matter tracts are replaced with a jumbled, disorganized texture that lacks the characteristic structure of gyri and sulci. The interhemispheric fissure in the posterior region is distorted and appears duplicated or "braided," which is anatomically impossible. The grey-white matter differentiation is lost within the affected area, replaced by a chaotic and implausible pattern. This violates the fundamental structural logic of the brain</evidence>. 
Based on these findings, I conclude that: 
<conclusion>The presence of anatomically impossible structures, such as the distorted interhemispheric fissure and the complete breakdown of normal gyral and white matter architecture, confirms that this image is a deepfake</conclusion>. 
</think>
```

Detailed guidelines and samples are available in the `data/` and `dataset_tools/` folders.

## Citation
Please refer to the paper for citation details.

---
Anonymized Release for Review.
# ACL2026-MedForge
