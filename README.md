# Latent Consistency Model Progressive Rendering

This is a prototype for demonstrating progressive rendering for [Latent Consistency Models](https://arxiv.org/abs/2310.04378). We implement progressive rendering by running several inference runs (one for each numbrer of steps in the range [1, 4]). This enables the previewing of the diffusion model's outputs in real time (1 sample per second).

We combine the fast inference capabilities of LMCs, which require on the order of 4 diffusion steps to generate high quality samples. LMCs trade off time for generation fidelity: more sample steps generate higher quality samples, but fewer sample steps take less time. We run each of these sampling processes in parallel and show each of the samples as they are generated. Further, because of the self-consistency constraint of Latent Consistency Model's samples with the same prompt and initial latent have the same high level structure. This gives the effect of the composition becoming progressively more detailed as more sample steps are taken.

## Installation

```bash
    pip install -r requirements.txt
```

## Usage

You can test out the prototype on an NVIDIA GPU by running the following command:

```bash
    python main.py
```
