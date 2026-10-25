"""Farklı süper çözünürlük modellerini aynı görüntü üzerinde çalıştırıp kalite ve hız açısından karşılaştırır."""

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader


# =============================================================================
# Ayarlar
# =============================================================================
# Not: Her model için mimariyi elle yazmak yerine spandrel kütüphanesini
# kullanıyorum. .pth dosyasına bakıp hangi mimari olduğunu (ESRGAN, SwinIR,
# SRVGGNetCompact...) kendisi anlıyor ve modeli kuruyor. Yeni bir model
# denemek için aşağıya bir satır eklemek yeterli.

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

MODELS = {
    # Şu an kullandığımız: küçük ve hızlı (~1.2M parametre)
    "general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesr-general-x4v3.pth"
    ),
    # Real-ESRGAN'ın asıl modeli: çok daha büyük (~16.7M parametre), daha yavaş
    "x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    ),
    # Aynı mimari ama GAN olmadan eğitilmiş: daha yumuşak, daha az uydurma
    "net-x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.1/RealESRNet_x4plus.pth"
    ),
    # Transformer tabanlı, gerçek fotoğraflar için eğitilmiş (orta boy)
    "swinir-m-real": (
        "https://github.com/JingyunLiang/SwinIR/releases/download/"
        "v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    ),
}


# =============================================================================
# Yardımcı fonksiyonlar
# =============================================================================

def pick_device() -> torch.device:
    # Not: CUDA'lı torch kuruluysa ekran kartını, değilse işlemciyi kullanıyor.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download(url: str) -> Path:
    """Model dosyasını indirir, zaten varsa tekrar indirmez."""
    path = MODELS_DIR / url.rsplit("/", 1)[-1]
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"    İndiriliyor: {path.name}")
        urllib.request.urlretrieve(url, path)
    return path


def load_model(path: Path, device: torch.device) -> ImageModelDescriptor:
    model = ModelLoader().load_from_file(path)
    # Not: spandrel görüntü dışı modelleri de yükleyebiliyor. Bu kontrol,
    # elimizdekinin gerçekten görüntüden görüntü üreten bir model olduğunu doğruluyor.
    assert isinstance(model, ImageModelDescriptor), f"{path.name} bir görüntü modeli değil"
    return model.to(device).eval()


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    # HWC, 0-255  ->  NCHW, 0-1
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


@torch.no_grad()
def upscale(model: ImageModelDescriptor, x: torch.Tensor) -> torch.Tensor:
    """Görüntüyü büyütür. Gerekirse kenarlara dolgu ekleyip sonra kırpar."""
    # Not: SwinIR gibi bazı modeller görüntüyü 8x8'lik pencerelere bölüyor,
    # bu yüzden genişlik ve yükseklik 8'in katı olmak zorunda. Kenarları
    # ayna yansımasıyla doldurup sonuçtan o kısmı geri kesiyorum.
    _, _, h, w = x.shape
    multiple = model.size_requirements.multiple_of
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    y = model(x)
    return y[..., : h * model.scale, : w * model.scale]


def timed_upscale(model: ImageModelDescriptor, x: torch.Tensor) -> tuple[torch.Tensor, float]:
    # Not: GPU işlemleri asenkron çalışıyor; synchronize çağırmazsam
    # süre ölçümü işlem bitmeden durur ve yanlış (çok kısa) çıkar.
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    y = upscale(model, x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return y, time.perf_counter() - start


# =============================================================================
# Ana akış
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Test görüntüsünün yolu")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS.keys(),
        default=list(MODELS.keys()),
        help="Denenecek modeller (varsayılan: hepsi)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        print(f"[!] Görüntü bulunamadı: {args.image}")
        return

    device = pick_device()
    image = Image.open(args.image).convert("RGB")
    x = image_to_tensor(image, device)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Cihaz: {device} | Görüntü: {image.width}x{image.height}\n")

    for name in args.models:
        print(f"[…] {name}")
        model = load_model(download(MODELS[name]), device)
        params = sum(p.numel() for p in model.model.parameters()) / 1e6

        y, elapsed = timed_upscale(model, x)
        out_path = RESULTS_DIR / f"{args.image.stem}_{name}.png"
        tensor_to_image(y).save(out_path)
        print(f"[✓] {model.architecture.name}, {params:.1f}M parametre, {elapsed:.2f} sn -> {out_path}")

        # Not: Bir sonraki modele geçmeden belleği boşaltıyorum,
        # yoksa büyük modellerde GPU belleği dolabiliyor.
        del model, y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Karşılaştırma için klasik yöntem
    bicubic = image.resize((image.width * 4, image.height * 4), Image.BICUBIC)
    bicubic.save(RESULTS_DIR / f"{args.image.stem}_bicubic.png")
    print(f"\n[✓] Bicubic karşılaştırma: {RESULTS_DIR / f'{args.image.stem}_bicubic.png'}")


if __name__ == "__main__":
    main()
