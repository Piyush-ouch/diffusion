# Stable Diffusion: Text-to-Image Generation and Analysis

A practical experiment using a pretrained **Stable Diffusion** pipeline to generate images from text prompts and study how inference settings affect generation quality, prompt alignment, and execution time.

## Project overview

This project uses the Hugging Face **Diffusers** library and the lightweight pretrained model [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd). It generates images from text prompts and compares three important factors:

1. **Inference steps** – the number of denoising iterations.
2. **Guidance scale (Classifier-Free Guidance / CFG)** – how strongly the generated image follows the text prompt.
3. **Sampling scheduler** – the numerical method used to update the noisy latent representation during denoising.

The supplied practical notebook is `Stable_Diffusion_Text_to_Image_Analysis.ipynb`.

## Objectives

- Generate images from natural-language prompts using a pretrained diffusion model.
- Understand the main components of a latent diffusion pipeline.
- Compare image-generation settings under controlled conditions.
- Evaluate generated images using quantitative metrics and visual inspection.
- Identify a practical balance between image quality and generation time.

## Technology stack

- Python
- PyTorch
- Hugging Face Diffusers
- Hugging Face Transformers
- CLIP (`openai/clip-vit-base-patch32`)
- NumPy, Pandas, SciPy
- Matplotlib
- Pillow

## How it works

The notebook uses a latent diffusion pipeline:

1. The text encoder converts the input prompt into embeddings.
2. The U-Net predicts noise in the latent representation during each denoising step.
3. A scheduler determines how the latent is updated at each step.
4. The VAE decoder converts the final latent representation into an image.

## Experiments

### 1. Inference-step comparison

The notebook tests **5, 10, 15, 20, 30, and 50 steps**, using a 100-step output as a reference for convergence analysis.

### 2. Guidance-scale comparison

The notebook tests CFG values **1, 3, 5, 7.5, 10, 15, and 20** to examine prompt adherence, saturation, sharpness, and diversity.

### 3. Scheduler comparison

The notebook compares the following schedulers:

- PNDM
- DDIM
- Euler
- Euler Ancestral
- LMS
- Heun
- DPM-Solver++ 2M
- DPM-Solver++ 2M with Karras sigmas
- UniPC

The schedulers are tested at 25 steps and 10 steps.

### 4. Generalisation check

Additional prompts cover a portrait, a landscape, and an object to check whether the observed parameter trends also appear with different subjects.

## Evaluation metrics

| Metric | Purpose |
|---|---|
| CLIP score | Estimates similarity between the prompt and generated image; higher generally indicates stronger alignment. |
| Sharpness | Uses variance of the Laplacian as a proxy for fine detail. |
| Saturation | Measures average HSV saturation and helps identify over-saturated outputs. |
| MSE vs. reference | Measures pixel-level difference from a high-step reference image; lower means closer to that reference. |
| Generation time | Measures the time required to generate an image; lower is faster. |

These metrics are proxies. They should be considered alongside visual inspection rather than treated as definitive measures of image quality.

## Setup and execution

### Option A: Google Colab (recommended)

1. Upload `Stable_Diffusion_Text_to_Image_Analysis.ipynb` to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another compatible GPU).
3. Run the notebook cells from top to bottom.
4. Allow the pretrained model and CLIP model to download when prompted.

### Option B: Local Python environment

Use Python with a compatible PyTorch installation. Install the notebook dependencies:

```bash
pip install --upgrade diffusers transformers accelerate safetensors scipy pandas numpy matplotlib pillow
```

Install the appropriate PyTorch build for your CPU/GPU from the [official PyTorch installation page](https://pytorch.org/get-started/locally/), then open the notebook in Jupyter.

> A CUDA-capable GPU is strongly recommended. CPU execution may be very slow. Model downloads require an internet connection the first time.

## Example prompt

```text
A realistic smart agricultural farm, green crops, automated irrigation systems,
drones monitoring the fields, clear blue sky, high quality
```

The notebook uses a fixed random seed for controlled comparisons so that changes in output are more attributable to the parameter being tested.

## Results and generated files

At the end, the notebook saves the following files in a `results/` directory:

```text
results/
├── exp1_steps.csv
├── exp2_guidance.csv
├── exp3_schedulers_25steps.csv
├── exp3_schedulers_10steps.csv
└── baseline.png
```

The CSV files contain experiment measurements. The baseline image is the generated example image. Plots and image grids are displayed during notebook execution.

## Main observations

The notebook's written analysis describes these general trends:

- Very few inference steps can produce blurry or incompletely denoised images; quality gains tend to diminish as the step count increases.
- Moderate guidance generally balances prompt alignment and natural-looking output. Excessively high guidance can cause saturation and artefacts.
- Higher-order schedulers such as DPM++ 2M, DPM++ 2M Karras, and UniPC are described as producing useful results with fewer steps.
- A fixed seed and multiple evaluation measures make comparisons more meaningful.

The exact metric values depend on the runtime, model version, and generated outputs. Refer to the CSV files produced by your own notebook run for measured results.

## Limitations

- The project uses the lightweight `segmind/tiny-sd` model by default; output quality may differ from larger Stable Diffusion models.
- CLIP score, sharpness, saturation, and MSE are imperfect proxies for human-perceived quality.
- The experiments use a limited set of prompts and random seeds.
- Results can vary with hardware, library versions, and model revisions.

## Repository contents

The practical work is provided as a Jupyter notebook:

- `Stable_Diffusion_Text_to_Image_Analysis.ipynb` – setup, image generation, experiments, evaluation, visualisations, and CSV export.

If the notebook is stored under a different filename or directory in this repository, update this entry to match the repository.

## Author

**Piyush Jangade**  
**PRN:** 202401110040

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Segmind Tiny Stable Diffusion model](https://huggingface.co/segmind/tiny-sd)
- [CLIP model – OpenAI / Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
