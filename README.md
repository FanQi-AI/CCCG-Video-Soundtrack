## Customized Condition Controllable Generation for Video Soundtrack（Accepted by CVPR 2025）

## Abstract
Recent advancements in latent diffusion models (LDMs) have led to innovative approaches in music generation, allowing for increased flexibility and integration with other modalities. However, existing methods often rely on a two-step process that fails to capture the artistic essence of videos, particularly in the context of complex videos requiring detailed sound effect and diverse instrumentation. In this paper, we propose a novel framework for generating video soundtracks that simultaneously produces music and sound effect tailored to the video content. Our method incorporates a Contrastive Visual-Sound-Music pretraining process that maps these modalities into a unified feature space, enhancing the model's ability to capture intricate audio dynamics. We design Spectrum Divergence Masked Attention for Unet to differentiate between the unique characteristics of sound effect and music. We utilize Score-guided Noise Iterative Optimization to provide musicians with customizable control during the generation process. Extensive evaluations on the FilmScoreDB and SymMV\&HIMV datasets demonstrate that our approach significantly outperforms state-of-the-art baselines in both subjective and objective assessments, highlighting its potential as a robust tool for video soundtrack generation.
## 1. Installation

``` shell
conda create -n CCCGLDM python=3.9
conda activate CCCGLDM

cd RLScale-Lora
pip install -r requirements.txt
```

## 2. Training

### Preparations

1. We use [VideoCLIP](https://github.com/CryhanFang/CLIP2Video) to extract the video feature, use [CLAP](https://github.com/LAION-AI/CLAP) to to extract the music feature。then use [AudioMAE](https://github.com/facebookresearch/AudioMAE) as the sound encoder based [Audioldm2](https://github.com/haoheliu/audioldm2)

### Commands

```shell
python train.py
```

## 3. Inference

```shell
python infer_musicldm.py
```
## 4. Acknowledgements

We sincerely thank the following repositories and their authors for providing valuable references and inspiration:

\- [AudioLDM] (https://github.com/haoheliu/AudioLDM):

\- [AudioLDM2] (https://github.com/haoheliu/audioldm2):

\- [MusicLDM] (https://github.com/RetroCirce/MusicLDM):

Their work has greatly contributed to our research and implementation.



