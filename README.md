# MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning

Official code release for our **ACL 2026 (Main Conference)** paper: *MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning* ([arXiv:2603.18577](https://arxiv.org/abs/2603.18577)).

This repository contains **MedForge-Reasoner** training and evaluation code: core training scripts (SFT / Forgery-aware GSPO), evaluation utilities, and dataset tooling described in the paper. For full benchmark data, model weights, and an interactive demo, please use the Hugging Face resources below and comply with their licenses and terms of use.

## Hugging Face resources

| Resource | Link | Notes |
|----------|------|--------|
| **MedForge-90K** (full dataset) | [RichardChenZH/MedForge-90K](https://huggingface.co/datasets/RichardChenZH/MedForge-90K) | Released under the dataset card license (**C-UDA** on the Hub); use only as permitted by that license and any accompanying documentation. |
| **MedForge-Reasoner** (weights) | [RichardChenZH/MedForge-Reasoner](https://huggingface.co/RichardChenZH/MedForge-Reasoner) | Main experimental checkpoint; see the model card for the license and intended use. |
| **Interactive demo** | [medforge-medical-deepfake-detector](https://huggingface.co/spaces/RichardChenZH/medforge-medical-deepfake-detector) | Try the model in-browser via a Gradio Space. |

We welcome the community to **try the demo**, **download the dataset and weights for research**, and **follow the open licenses and ethical use guidelines** stated on each Hugging Face page. This software is for **research only**; it is not a clinical product.

Arxiv Paper Link: https://arxiv.org/abs/2603.18577

## Abstract

Text-guided image editors can now manipulate authentic medical scans with high fidelity, threatening clinical trust and safety. MedForge provides a pre-hoc, evidence-grounded solution for medical forgery detection. We introduce **MedForge-90K**, a large-scale benchmark of realistic lesion edits with expert-guided reasoning supervision. **MedForge-Reasoner** performs localize-then-analyze reasoning, predicting suspicious regions before producing a verdict, and is aligned via Forgery-aware GSPO to strengthen grounding and reduce hallucinations. Full definitions, training details, and experiments are in the [paper](https://arxiv.org/abs/2603.18577).

## Project structure

- `medforge_reasoner/`: Core logic for training (SFT/GSPO) and inference. Includes the Forgery-aware reward plugin.
- `data/`: Optional small local examples (if present) for running scripts; **not** a substitute for the full benchmark—use [MedForge-90K on Hugging Face](https://huggingface.co/datasets/RichardChenZH/MedForge-90K) for the released dataset.
- `evaluation/`: Scripts for LLM-as-judge evaluation and benchmark prompt generation.
- `dataset_tools/`: Tools used for generating high-fidelity forgeries and expert-guided annotations.
- `ms-swift/`: (To be installed) ModelScope SWIFT framework for efficient fine-tuning.

## Environment setup

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

### Supervised fine-tuning (SFT)

```bash
cd medforge_reasoner
bash train.sh
```

### Training (GSPO)

```bash
cd medforge_reasoner
bash train_gspo.sh
```

### Evaluation

The evaluation is divided into two primary stages: **Detection** and **Explainability**.

1. **Detection evaluation (core metrics)**:
   This stage evaluates the model's ability to classify images as real or forgery and its localization accuracy. It computes classification Accuracy, F1 Score, and Mean Intersection over Union (mIoU) for bounding boxes.

   ```bash
   cd evaluation/detection_eval
   python eval.py --input /path/to/inference_results.jsonl --output ./results
   ```

2. **Explainability evaluation (reasoning quality)**:
   This stage uses an LLM-as-judge approach to evaluate the quality and grounding of the model's generated reasoning (Chain-of-Thought).

   ```bash
   cd evaluation/explainability_llm_as_judge/gemini3pro-judge
   python judge_main.py
   ```

## Dataset: MedForge-90K

According to the paper ([arXiv:2603.18577](https://arxiv.org/abs/2603.18577)), **MedForge-90K** contains:

- **30K real images** (chest X-ray, brain MRI, fundus)
- **30K lesion implant forgeries**
- **30K lesion removal forgeries**

The **canonical open release** of the full benchmark is **[RichardChenZH/MedForge-90K](https://huggingface.co/datasets/RichardChenZH/MedForge-90K)**. Please download and use it from the Hub, and respect the **dataset license (C-UDA)** and any usage restrictions described there.

If this repository includes a local `data/` tree, treat it as **optional illustration** for tooling or paths in scripts; for replication and benchmarking, use the Hugging Face dataset above.

### Dataset CoT example

The following is an example of the structured Chain-of-Thought (CoT) used for expert-guided reasoning (format as in the paper):

```xml
<redacted_thinking>
Okay, let's see. I am starting to analyze whether this image is a deepfake: 
<description>This is an axial T2-weighted MRI of a brain</description>. 
I have identified the suspect deepfake area at: <|object_ref_start|>"deepfake"<|object_ref_end|><|box_start|>x1="350" y1="452" x2="596" y2="761"<|box_end|>. 
Now I am going to perform a detailed analysis of this suspect region. 
In my observation, the evidence for deepfake is: 
<evidence>The deepfake region displays a significant disruption of normal brain anatomy. The white matter tracts are replaced with a jumbled, disorganized texture that lacks the characteristic structure of gyri and sulci. The interhemispheric fissure in the posterior region is distorted and appears duplicated or "braided," which is anatomically impossible. The grey-white matter differentiation is lost within the affected area, replaced by a chaotic and implausible pattern. This violates the fundamental structural logic of the brain</evidence>. 
Based on these findings, I conclude that: 
<conclusion>The presence of anatomically impossible structures, such as the distorted interhemispheric fissure and the complete breakdown of normal gyral and white matter architecture, confirms that this image is a deepfake</conclusion>. 
</redacted_thinking>
```

Detailed annotation guidelines are under `dataset_tools/` (e.g. guideline documents) when present in your checkout.

## License

Source code in this repository is licensed under the **Apache License, Version 2.0** — see the [`LICENSE`](LICENSE) file. **MedForge-90K**, **MedForge-Reasoner** weights, and other materials on Hugging Face remain under the licenses and terms stated on their respective Hub pages (for example C-UDA for the dataset); this repository license applies only to the code here.

## Citation

If you use this code, the dataset, or the model, please cite:

```bibtex
@misc{chen2026medforgeinterpretablemedicaldeepfake,
      title={MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning}, 
      author={Zhihui Chen and Kai He and Qingyuan Lei and Bin Pu and Jian Zhang and Yuling Xu and Mengling Feng},
      year={2026},
      eprint={2603.18577},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.18577}, 
}
```

**Reference:** Zhihui Chen, Kai He, Qingyuan Lei, Bin Pu, Jian Zhang, Yuling Xu, Mengling Feng. *MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning.* ACL 2026 (Main). arXiv:2603.18577. [https://arxiv.org/abs/2603.18577](https://arxiv.org/abs/2603.18577)
