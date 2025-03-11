# Gull: Generative Multifunctional Neural Audio Codec

An unofficial implementation of Gull, a generative multifunctional neural audio codec operating in the frequency domain.

<p align="left">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Gull-Codec-Training" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/JusperLee/Gull-Codec-Training?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

## 📢 News

- 🚀 **2025.03**: First release of our unofficial Gull implementation!

## 🛠️ Installation Env

```bash
# Clone the repository
git clone https://github.com/JusperLee/Gull-Codec-Training.git
cd Gull-Codec-Training

# Install dependencies
pip install torch torchaudio torchaudio
pip install -r requirements.txt

```

## 📚 Usage

### Training

```bash
python train.py --conf_dir=configs/gull.yml
```

## 📊 Experimental Results

> **Note**: Currently in the training process

## 📄 References

This implementation is based on the following paper:

```
@article{luo2024gull,
title={Gull: A Generative Multifunctional Audio Codec},
author={Luo, Yi and Yu, Jianwei and Chen, Hangting and Gu Rongzhi and Weng, Chao},
journal={arXiv preprint arXiv:2404.04947},
year={2024}
}
```

Training and data processing code are based on the following repository:

- Training code: [Apollo](https://github.com/JusperLee/Apollo)
- Data processing code: [Apollo-data-preprocess](https://github.com/JusperLee/Apollo-data-preprocess)

## ⚖️ License

This unofficial implementation is released under the [Apache-2.0 license](LICENSE).

## 🙏 Disclaimer

This is an unofficial implementation based on published research. We are not affiliated with the original authors of the Gull paper. This code is provided for research and educational purposes only.