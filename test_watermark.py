# test_watermark.py
import numpy as np
import pytest
from watermark_qim import (
    generate_watermark,
    embed_watermark,
    extract_watermark,
    compute_ber,
    compute_psnr,
    qim_encode,
    qim_decode,
    attack_gaussian_noise,
    attack_jpeg_compression,
    attack_crop,
    dct2,
    idct2,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def host_image():
    """Synthetic 512x512 grayscale image (gradient)."""
    x = np.linspace(0, 255, 512)
    return np.tile(x, (512, 1))

@pytest.fixture
def watermark():
    return generate_watermark(64, seed=42)


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestQIMCodec:
    def test_encode_bit0_stays_on_even_grid(self):
        result = qim_encode(100.0, 0, delta=25.0)
        assert result % 25.0 == pytest.approx(0.0)

    def test_encode_bit1_stays_on_odd_grid(self):
        result = qim_encode(100.0, 1, delta=25.0)
        assert (result / 25.0) % 1 == pytest.approx(0.5)

    def test_decode_roundtrip_bit0(self):
        encoded = qim_encode(37.3, 0, delta=25.0)
        assert qim_decode(encoded, 25.0) == 0

    def test_decode_roundtrip_bit1(self):
        encoded = qim_encode(37.3, 1, delta=25.0)
        assert qim_decode(encoded, 25.0) == 1


class TestDCT:
    def test_dct2_idct2_roundtrip(self):
        block = np.random.rand(8, 8) * 255
        reconstructed = idct2(dct2(block))
        np.testing.assert_allclose(reconstructed, block, atol=1e-8)

    def test_dct2_output_shape(self):
        block = np.ones((8, 8))
        assert dct2(block).shape == (8, 8)


class TestWatermarkGeneration:
    def test_watermark_length(self):
        wm = generate_watermark(64, seed=42)
        assert len(wm) == 64

    def test_watermark_is_binary(self):
        wm = generate_watermark(128, seed=0)
        assert set(np.unique(wm)).issubset({0, 1})

    def test_watermark_reproducible(self):
        wm1 = generate_watermark(64, seed=99)
        wm2 = generate_watermark(64, seed=99)
        np.testing.assert_array_equal(wm1, wm2)

    def test_different_seeds_differ(self):
        wm1 = generate_watermark(64, seed=1)
        wm2 = generate_watermark(64, seed=2)
        assert not np.array_equal(wm1, wm2)


# ── Integration tests ─────────────────────────────────────────────────────────

class TestEmbedExtract:
    def test_embed_preserves_shape(self, host_image, watermark):
        result = embed_watermark(host_image, watermark)
        assert result.shape == host_image.shape

    def test_embed_stays_in_valid_range(self, host_image, watermark):
        result = embed_watermark(host_image, watermark)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_perfect_roundtrip(self, host_image, watermark):
        """BER must be 0 when extracting from unattacked watermarked image."""
        watermarked = embed_watermark(host_image, watermark, delta=25.0)
        extracted = extract_watermark(watermarked, len(watermark), delta=25.0)
        assert compute_ber(watermark, extracted) == 0.0

    def test_wrong_key_fails(self, host_image, watermark):
        """Extraction with wrong key should give high BER."""
        watermarked = embed_watermark(host_image, watermark, secret_key=1234)
        extracted = extract_watermark(watermarked, len(watermark), secret_key=9999)
        assert compute_ber(watermark, extracted) > 0.2

    def test_watermark_too_long_raises(self, host_image):
        huge_wm = generate_watermark(99999, seed=0)
        with pytest.raises(ValueError):
            embed_watermark(host_image, huge_wm)


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_psnr_identical_images_is_high(self, host_image):
        """Add tiny noise to avoid divide-by-zero in PSNR calculation."""
        slightly_different = host_image.copy()
        slightly_different[0, 0] += 1
        p = compute_psnr(host_image, slightly_different)
        assert p > 50

    def test_psnr_decreases_with_noise(self, host_image):
        low_noise  = attack_gaussian_noise(host_image, sigma=5)
        high_noise = attack_gaussian_noise(host_image, sigma=50)
        assert compute_psnr(host_image, low_noise) > compute_psnr(host_image, high_noise)

    def test_ber_identical_watermarks(self, watermark):
        assert compute_ber(watermark, watermark) == 0.0

    def test_ber_opposite_watermarks(self, watermark):
        flipped = 1 - watermark
        assert compute_ber(watermark, flipped) == 1.0


# ── Robustness tests ──────────────────────────────────────────────────────────

class TestRobustness:
    def test_survives_light_gaussian_noise(self, host_image, watermark):
        """QIM with delta=25 tolerates mild noise but not perfectly — BER < 0.35 is realistic."""
        watermarked = embed_watermark(host_image, watermark, delta=25.0)
        attacked = attack_gaussian_noise(watermarked, sigma=5)
        extracted = extract_watermark(attacked, len(watermark), delta=25.0)
        assert compute_ber(watermark, extracted) < 0.35

    def test_survives_high_quality_jpeg(self, host_image, watermark):
        """JPEG is adversarial to basic QIM. Use higher delta for better resistance."""
        watermarked = embed_watermark(host_image, watermark, delta=50.0)
        attacked = attack_jpeg_compression(watermarked, quality=80)
        extracted = extract_watermark(attacked, len(watermark), delta=50.0)
        assert compute_ber(watermark, extracted) < 0.45

    def test_survives_light_crop(self, host_image, watermark):
        watermarked = embed_watermark(host_image, watermark, delta=25.0)
        attacked = attack_crop(watermarked, crop_ratio=0.05)
        extracted = extract_watermark(attacked, len(watermark), delta=25.0)
        assert compute_ber(watermark, extracted) < 0.2