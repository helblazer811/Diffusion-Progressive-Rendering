# Progressive Rendering for Diffusion

This is a prototype for demonstrating progressive rendering for Diffusion and [Latent Consistency Models](https://arxiv.org/abs/2310.04378). We implement progressive rendering by running several inference runs (one for each number of steps in the range [1, 4]) simultaneously in parallel. This enables the previewing of the diffusion model's outputs in real-time (1 sample per second). 

https://github.com/helblazer811/LCM-Progressive-Rendering/assets/14181830/eb2889c8-276c-4802-94b9-7bf78c019dd4

This demonstration applies to both Diffusion models and Latent Consistency Models, however, we demonstrate it with Latent Consistency Models. We combine the fast inference capabilities of LMCs, which require on the order of 4 diffusion steps to generate high-quality samples. LMCs trade-off time for generation fidelity: more sample steps generate higher quality samples, but fewer sample steps take less time. We run each of these sampling processes in parallel and show each of the samples as they are generated. Further, because of the self-consistency constraint of Latent Consistency Model's samples with the same prompt and initial latent have the same high-level structure. This gives the effect of the composition becoming progressively more detailed as more sample steps are taken. It is worth noting that there is not a need to run parallel inference for Latent Consistency Models because of their multistep scheduler, but it is necessary for diffusion models. 

## Installation

```bash
    pip install -r requirements.txt
```

## Usage

You can test out the prototype on an NVIDIA GPU by running the following command:

```bash
    python main.py
```
