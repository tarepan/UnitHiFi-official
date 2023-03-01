<div align="center">

# UnitHiFi : Speech Resynthesis from unit with HiFi-GAN <!-- omit in toc -->
[![OpenInColab]][notebook]
[![paper_badge]][paper]

</div>

Clone of the official ***UnitHiFi*** implementation.  

[paper]: https://arxiv.org/abs/2104.00355
[paper_badge]: http://img.shields.io/badge/paper-arxiv.2104.00355-B31B1B.svg
[notebook]: https://colab.research.google.com/github/tarepan/UnitHiFi-official/blob/main/unithifi.ipynb
[OpenInColab]: https://colab.research.google.com/assets/colab-badge.svg

<p align="center"><img width="70%" src="img/fig.png" /></p>

## Quick Links
- [Samples](https://speechbot.github.io/resynthesis/index.html)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)

## Setup

### Software
Requirements:
* Python >= 3.6
* PyTorch v1.8
* Install dependencies
    ```bash
    git clone https://github.com/tarepan/UnitHiFi-official.git
    cd UnitHiFi-official
    pip install -r requirements.txt
    ```
### Data

#### For LJSpeech:
1. Download LJSpeech dataset from [here](https://keithito.com/LJ-Speech-Dataset/) into ```data/LJSpeech-1.1``` folder.
2. Downsample audio from 22.05 kHz to 16 kHz and pad
   ```
   bash
   python ./scripts/preprocess.py \
   --srcdir data/LJSpeech-1.1/wavs \
   --outdir data/LJSpeech-1.1/wavs_16khz \
   --pad
   ```

#### For VCTK:
1. Download VCTK dataset through `speechcorpusy`, move contents under ```data/```.
   ```python
   from speechcorpusy import load_preset
   corpus = load_preset("VCTK", root="<your_root>", download=False).get_contents()
   ```
   ```bash
   mkdir "data"
   mv "tmp/corpuses/VCTK/ver0_92/contents/VCTK-Corpus" "data"
   ```
2. Downsample audio to 16 kHz, trim trailing silences and pad
   ```bash
   python ./scripts/preprocess.py \
   --srcdir "data/VCTK-Corpus" \
   --outdir "data/VCTK-Corpus/wav16_silence_trimmed_padded" \
   --trim --pad
   ```

## Training

For training from scratch, you needs below steps:  

1. Encoder<sub>content</sub> training
2. z_c extraction
3. Encoder<sub>f<sub>o</sub></sub> training
4. z_<sub>f<sub>o</sub></sub> extraction
5. Decoder training

This repository provides some shortcuts. In minimum case, you needs to execute only step 3 and 5.  

Currently, we support the following training schemes:

| Dataset  | Encoder<sub>c</sub> SSL | Dictionary Size | Config Path                               |
| -------- |------------------------ | --------------- | ------------------------------------------|
| LJSpeech | HuBERT                  | 100             | ```configs/LJSpeech/hubert100_lut.json``` |
| LJSpeech | CPC                     | 100             | ```configs/LJSpeech/cpc100_lut.json```    |
| VCTK     | HuBERT                  | 100             | ```configs/VCTK/hubert100_lut.json```     |
| VCTK     | CPC                     | 100             | ```configs/VCTK/cpc100_lut.json```        |

### 1. Encoder<sub>content</sub> Training
It's tough work, but you can try SSL by yourself.  
Fortunately, pretrained CPC and HuBERT is provided. You can skip this step.  

### 2. z_c Extraction
For LJSpeech or VCTK, we provide extracted z_c! In this case, you can skip this step.  

If you hope to encode other datasets, follow the instructions described in the [GSLM code](https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm).  

To parse CPC output:
```bash
python scripts/parse_cpc_codes.py \
--manifest cpc_output_file \
--wav-root wav_root_dir \
--outdir parsed_cpc
```

To parse HuBERT output:
```bash
python parse_hubert_codes.py \
--codes hubert_output_file \
--manifest hubert_tsv_file \
--outdir parsed_hubert 
```

### 3. Encoder<sub>f<sub>o</sub></sub> training
Train f<sub>o</sub> VQVAE for Encoder<sub>f<sub>o</sub></sub>:
```bash
python train_f0_vq.py \
--checkpoint_path checkpoints/lj_f0_vq \
--config configs/LJSpeech/f0_vqvae.json
```
### 4. z_<sub>f<sub>o</sub></sub> extraction
This step is automatically executed in step 5.  

### 5. Decoder training
The final step is below:
```bash
python train.py \
--checkpoint_path checkpoints/lj_vqvae \
--config configs/LJSpeech/vqvae256_lut.json
```
 
## Inference
To generate, simply run:
```bash
python inference.py \
--checkpoint_file checkpoints/vctk_cpc100 \
-n 10 \
--output_dir generations
```

To synthesize multiple speakers:
```bash
python inference.py \
--checkpoint_file checkpoints/vctk_cpc100 \
-n 10 \
--vc \
--input_code_file datasets/VCTK/cpc100/test.txt \
--output_dir generations_multispkr
```

You can also generate with codes from a different dataset:
```bash
python inference.py \
--checkpoint_file checkpoints/lj_cpc100 \
-n 10 \
--input_code_file datasets/VCTK/cpc100/test.txt \
--output_dir generations_vctk_to_lj
```

## Difference from the Paper
- Encoder<sub>content</sub>=VQVAE is disabled (original repository supports this)
- Encoder<sub>speaker</sub> is just Embedding, not LSTM

## Citation
```
@inproceedings{polyak21_interspeech,
  author={Adam Polyak and Yossi Adi and Jade Copet and 
          Eugene Kharitonov and Kushal Lakhotia and 
          Wei-Ning Hsu and Abdelrahman Mohamed and Emmanuel Dupoux},
  title={{Speech Resynthesis from Discrete Disentangled Self-Supervised Representations}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
}
``` 

## Acknowledgements
This implementation uses code from the following repos:  

- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Jukebox](https://github.com/openai/jukebox)
