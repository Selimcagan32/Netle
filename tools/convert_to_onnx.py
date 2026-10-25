"""realesr-general-x4v3 modelini indirir, ONNX'e (FP32 + FP16) çevirir ve isteğe bağlı olarak bir görüntüde test eder."""

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxconverter_common import float16
from PIL import Image


# =============================================================================
# Ayarlar
# =============================================================================
# Not: Sabitleri en üstte topladım. Bir şeyi değiştirmek istediğimde
# kodun içinde aramak zorunda kalmıyorum.

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth"
)
SCALE = 4  # Model görüntüyü her kenarda 4 kat büyütüyor (500x500 -> 2000x2000)

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "realesr-general-x4v3.pth"
ONNX_FP32_PATH = MODELS_DIR / "realesr-general-x4v3.onnx"
ONNX_FP16_PATH = MODELS_DIR / "realesr-general-x4v3-fp16.onnx"

# PyTorch ile ONNX çıktısı arasındaki kabul edilebilir en büyük fark.
# Not: Sıfır olmasını beklememek lazım; işlemlerin sırası değişince
# float hesaplarda çok küçük yuvarlama farkları oluşuyor.
MAX_ALLOWED_DIFF = 1e-3


# =============================================================================
# Model mimarisi
# =============================================================================
# Not: .pth dosyası sadece ağırlıkları (sayıları) içeriyor, mimariyi değil.
# Ağırlıkları yükleyebilmek için mimariyi birebir aynı şekilde tanımlamam
# gerekiyor. Orijinali basicsr paketinde ama o paket yeni torchvision
# sürümleriyle import hatası veriyor, bu yüzden burada kendim yazdım.

class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN'ın küçük ve hızlı süper çözünürlük ağı."""

    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_convs=32, scale=SCALE):
        super().__init__()
        self.scale = scale

        # Dikkat: Değişken adı "body" olmak zorunda. Ağırlık dosyasındaki
        # anahtarlar "body.0.weight", "body.1.weight"... şeklinde. Adı
        # değiştirirsem load_state_dict eşleştiremez ve hata verir.
        self.body = nn.ModuleList()

        # İlk katman: 3 kanallı RGB görüntüyü 64 özellik kanalına çeviriyor.
        # Conv2d(giriş, çıkış, kernel=3, stride=1, padding=1)
        # Not: 3x3 kernel + padding 1 = görüntünün boyutu değişmiyor.
        self.body.append(nn.Conv2d(in_channels, num_features, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_features))

        # Gövde: 32 tane conv + aktivasyon. Asıl "öğrenme" burada.
        # Not: PReLU, ReLU'nun negatif değerleri tamamen sıfırlamayan hali.
        # Negatif kısmın eğimi öğreniliyor, hem de her kanal için ayrı ayrı.
        for _ in range(num_convs):
            self.body.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_features))

        # Son katman: 3 * 4 * 4 = 48 kanal üretiyor. Neden 48?
        # Çünkü PixelShuffle bu 48 kanalı alıp her pikseli 4x4'lük bir
        # bloğa açıyor ve sonuçta 3 kanallı, 4 kat büyük bir görüntü çıkıyor.
        # Not: Buna "sub-pixel convolution" deniyor. Büyütmeyi en sonda
        # yaptığı için tüm hesaplama küçük görüntü üzerinde yapılıyor, bu da
        # modeli hızlı yapıyor.
        self.body.append(nn.Conv2d(num_features, out_channels * scale * scale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)

        # Not: Model, büyütülmüş görüntünün tamamını değil, basitçe büyütülmüş
        # (nearest) görüntüye eklenecek "farkı" öğreniyor. Buna residual
        # learning deniyor. Sıfırdan görüntü üretmekten çok daha kolay
        # olduğu için eğitim daha hızlı ve kararlı oluyor.
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return out + base


# =============================================================================
# Yardımcı fonksiyonlar
# =============================================================================

def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def download_weights(url: str, destination: Path) -> None:
    """Ağırlık dosyasını indirir. Zaten varsa tekrar indirmez."""
    if destination.exists():
        print(f"[✓] Ağırlıklar zaten var: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("[…] Ağırlıklar indiriliyor...")
    urllib.request.urlretrieve(url, destination)
    print(f"[✓] İndirildi: {destination} ({file_size_mb(destination):.1f} MB)")


def load_model(weights_path: Path) -> nn.Module:
    """Mimariyi kurar, ağırlıkları yükler ve modeli tahmin moduna alır."""
    model = SRVGGNetCompact()

    # Not: weights_only=True güvenlik için önemli. .pth dosyaları pickle
    # formatında ve pickle içinde çalıştırılabilir kod bulunabilir. Bu
    # ayar sadece tensörlerin yüklenmesine izin veriyor.
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Real-ESRGAN ağırlıkları {"params": {...}} şeklinde saklanıyor
    state_dict = checkpoint.get("params", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    # Not: eval() unutulmamalı. Eğitime özel davranışları (dropout,
    # batchnorm güncellemesi gibi) kapatıyor. Bu modelde ikisi de yok
    # ama alışkanlık olarak her zaman çağırıyorum.
    model.eval()
    return model


# =============================================================================
# ONNX'e dönüştürme
# =============================================================================
# Not: ONNX, modeli framework'ten bağımsız bir formatta saklıyor. Böylece
# PyTorch ile eğitilmiş modeli tarayıcıda ONNX Runtime Web ile
# çalıştırabiliyorum. Tarayıcıda PyTorch yok.

def export_fp32(model: nn.Module, output_path: Path) -> None:
    """Modeli 32 bitlik (tam hassasiyet) ONNX dosyası olarak kaydeder."""
    # Not: Exporter, modelin içinden örnek bir girdi geçirip hangi
    # işlemlerin yapıldığını kaydediyor. Girdinin içeriği önemli değil,
    # sadece şekli (1 görüntü, 3 kanal, yükseklik, genişlik) önemli.
    dummy_input = torch.rand(1, 3, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        # Not: Bunu yazmasaydım model sadece 64x64 görüntü kabul ederdi.
        # Yükseklik (2) ve genişlik (3) boyutlarını değişken yapıyorum.
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        },
        opset_version=17,
        dynamo=False,  # Klasik exporter, dynamic_axes ile sorunsuz çalışıyor
    )
    print(f"[✓] FP32 ONNX: {output_path} ({file_size_mb(output_path):.1f} MB)")


def convert_to_fp16(source_path: Path, output_path: Path) -> None:
    """FP32 modeli 16 bitlik hale getirir. Dosya boyutu yaklaşık yarıya iner."""
    # Not: FP16'da her sayı 4 yerine 2 byte tutuyor. Kalite kaybı bu tür
    # görüntü modellerinde gözle neredeyse fark edilmiyor. Tarayıcıda
    # indirme süresi ve GPU belleği açısından büyük kazanç.
    fp32_model = onnx.load(source_path)

    # Not: keep_io_types=True ile giriş ve çıkış float32 kalıyor, sadece
    # içerideki hesaplamalar float16 oluyor. Böylece tarayıcı tarafında
    # Float32Array ile çalışmaya devam edebiliyorum, dönüşümle uğraşmıyorum.
    fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
    onnx.save(fp16_model, output_path)
    print(f"[✓] FP16 ONNX: {output_path} ({file_size_mb(output_path):.1f} MB)")


def verify_export(model: nn.Module, onnx_path: Path) -> None:
    """Aynı girdiyi hem PyTorch hem ONNX modeline verip sonuçları karşılaştırır."""
    # Not: Dönüştürme hatasız bitti diye sonucun doğru olduğu anlamına
    # gelmiyor. Bu kontrol, iki modelin gerçekten aynı şeyi hesapladığını
    # gösteriyor. Bilerek 64x64'ten farklı bir boyut seçtim, böylece
    # dinamik boyutların da çalıştığını test etmiş oluyorum.
    test_input = torch.rand(1, 3, 48, 80)

    with torch.no_grad():  # Tahmin yaparken gradyan hesabına gerek yok
        expected = model(test_input).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": test_input.numpy()})[0]

    max_diff = np.abs(expected - actual).max()
    status = "✓" if max_diff < MAX_ALLOWED_DIFF else "!"
    print(f"[{status}] PyTorch–ONNX en büyük fark: {max_diff:.2e}")


# =============================================================================
# Görüntü üzerinde test
# =============================================================================

def image_to_tensor(image: Image.Image) -> np.ndarray:
    """PIL görüntüsünü modelin beklediği formata çevirir."""
    # Not: Modelin beklediği format ile görüntünün formatı farklı.
    #   Görüntü: (yükseklik, genişlik, kanal), değerler 0-255 arası tam sayı
    #   Model:   (batch, kanal, yükseklik, genişlik), değerler 0-1 arası ondalık
    # transpose(2, 0, 1): HWC -> CHW
    # [None]: başa batch boyutu ekliyor -> NCHW
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None]


def tensor_to_image(tensor: np.ndarray) -> Image.Image:
    """Model çıktısını tekrar görüntüye çevirir (image_to_tensor'ün tersi)."""
    # Not: Model bazen 0'ın altında ya da 1'in üstünde değerler üretebiliyor.
    # clip ile sınırlamazsam taşan pikseller yanlış renklerde çıkıyor.
    array = tensor[0].transpose(1, 2, 0)
    array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def upscale_with_onnx(onnx_path: Path, image: Image.Image) -> tuple[Image.Image, float]:
    """Görüntüyü ONNX modeliyle büyütür, sonucu ve geçen süreyi döndürür."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_tensor = image_to_tensor(image)

    start = time.perf_counter()
    output_tensor = session.run(None, {"input": input_tensor})[0]
    elapsed = time.perf_counter() - start

    return tensor_to_image(output_tensor), elapsed


def test_on_image(image_path: Path) -> None:
    """FP32, FP16 ve bicubic sonuçlarını yan yana karşılaştırmak için kaydeder."""
    image = Image.open(image_path).convert("RGB")  # PNG'lerde alfa kanalı olabiliyor
    width, height = image.size
    print(f"\n[…] Test görüntüsü: {width}x{height}")

    # Not: Şimdilik görüntüyü tek parça işliyorum. Büyük görüntülerde bellek
    # yetmeyebilir; bunu ileride tile (parçalara bölme) ile çözeceğim.
    if max(width, height) > 800:
        print("    Uyarı: Büyük görüntü, tile olmadan yavaş olabilir veya bellek yetmeyebilir.")

    for onnx_path in (ONNX_FP32_PATH, ONNX_FP16_PATH):
        result, elapsed = upscale_with_onnx(onnx_path, image)
        output_path = image_path.with_name(f"{image_path.stem}_{onnx_path.stem}.png")
        result.save(output_path)
        print(f"[✓] {onnx_path.name}: {elapsed:.2f} sn -> {output_path}")

    # Not: CPU'da FP16 genelde FP32'den hızlı değil, hatta yavaş olabiliyor.
    # Çünkü çoğu CPU'nun FP16 için özel donanımı yok. Burada amacım hızı
    # değil kaliteyi karşılaştırmak. FP16'nın asıl faydası tarayıcıda GPU'da.

    # Karşılaştırma için klasik yöntem: sadece komşu piksellerden tahmin
    bicubic = image.resize((width * SCALE, height * SCALE), Image.BICUBIC)
    bicubic_path = image_path.with_name(f"{image_path.stem}_bicubic.png")
    bicubic.save(bicubic_path)
    print(f"[✓] Bicubic karşılaştırma: {bicubic_path}")


# =============================================================================
# Ana akış
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",  # İsteğe bağlı: verilmezse sadece dönüştürme yapılır
        type=Path,
        help="Test için kullanılacak görüntünün yolu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    download_weights(MODEL_URL, WEIGHTS_PATH)
    model = load_model(WEIGHTS_PATH)

    export_fp32(model, ONNX_FP32_PATH)
    convert_to_fp16(ONNX_FP32_PATH, ONNX_FP16_PATH)
    verify_export(model, ONNX_FP32_PATH)

    if args.image:
        test_on_image(args.image)


# Not: Bu kontrol sayesinde dosya doğrudan çalıştırıldığında main() çalışıyor,
# ama başka bir dosyadan import edildiğinde çalışmıyor. Böylece ileride
# SRVGGNetCompact sınıfını başka bir yerde de kullanabilirim.
if __name__ == "__main__":
    main()
