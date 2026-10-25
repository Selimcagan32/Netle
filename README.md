# Netle

Browser-based AI image upscaler. *("Netle" means "sharpen it" in Turkish.)*

> 🚧 Work in progress

## About

Netle aims to upscale images 4× using Real-ESRGAN, running entirely in the browser with ONNX Runtime Web. No server needed, and images never leave the user's device.

## Project Structure

- `tools/` — Python scripts for preparing the model (download, ONNX conversion, testing)
- `web/` — Browser app (coming soon)

## Model Preparation

```bash
pip install torch onnx onnxruntime onnxconverter-common pillow numpy
python tools/convert_to_onnx.py              # download + convert
python tools/convert_to_onnx.py test.jpg     # download + convert + test on an image
```

## Credits

Model: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (`realesr-general-x4v3`) by Xintao Wang et al., BSD-3-Clause License.

---

## Türkçe

Netle, tarayıcıda çalışan yapay zeka destekli bir görüntü büyütme aracı. Real-ESRGAN modeliyle görüntüleri 4 kat büyütmeyi hedefliyor. Model ONNX Runtime Web ile tamamen tarayıcıda çalışıyor; sunucu gerekmiyor ve görüntüler kullanıcının cihazından çıkmıyor.