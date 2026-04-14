"""
changed a77: 2023-06-30
Mini-Projet : Sécurisation des images 2D par tatouage numérique basé sur QIM
=============================================================================
Implémentation complète du système de tatouage numérique (watermarking) en Python
utilisant la méthode QIM (Quantization Index Modulation) dans le domaine DCT 2D.

Architecture du système :
    1. Lecture image hôte
    2. Transformation fréquentielle (DCT 2D)
    3. Insertion watermark par QIM
    4. Simulation d'attaques
    5. Extraction et évaluation
    6. Reconstruction image tatouée
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1.  UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def load_image_gray(path: str) -> np.ndarray:
    """Charge une image et la convertit en niveaux de gris (float64)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    return img.astype(np.float64)


def generate_watermark(size: int, seed: int = 42) -> np.ndarray:
    """Génère un watermark binaire pseudo-aléatoire de longueur `size`."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=size).astype(np.int32)


def select_coefficients(n_coeff: int, block_size: int = 8, secret_key: int = 1234) -> list[tuple[int, int]]:
    """
    Sélectionne pseudo-aléatoirement des indices (i,j) de moyenne fréquence
    dans un bloc DCT 8x8 pour y insérer le watermark.
    """
    # Coordonnées de moyenne fréquence dans un bloc 8x8
    mid_freq = [
        (1, 2), (2, 1), (2, 2), (1, 3), (3, 1),
        (2, 3), (3, 2), (3, 3), (1, 4), (4, 1),
        (2, 4), (4, 2), (3, 4), (4, 3), (4, 4),
        (5, 1), (1, 5), (5, 2), (2, 5), (5, 3),
        (3, 5), (5, 4), (4, 5), (5, 5), (6, 1),
        (1, 6), (6, 2), (2, 6),
    ]
    random.seed(secret_key)
    selected = random.sample(mid_freq * ((n_coeff // len(mid_freq)) + 2), n_coeff)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DCT 2D PAR BLOCS
# ─────────────────────────────────────────────────────────────────────────────

def dct2(block: np.ndarray) -> np.ndarray:
    """DCT-II 2D sur un bloc."""
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(block: np.ndarray) -> np.ndarray:
    """DCT inverse 2D sur un bloc."""
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def apply_dct_blocks(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Applique la DCT 2D sur tous les blocs non-chevauchants de l'image."""
    h, w = image.shape
    dct_image = np.zeros_like(image)
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            dct_image[i:i+block_size, j:j+block_size] = dct2(image[i:i+block_size, j:j+block_size])
    return dct_image


def apply_idct_blocks(dct_image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Applique la DCT inverse 2D sur tous les blocs."""
    h, w = dct_image.shape
    image = np.zeros_like(dct_image)
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            image[i:i+block_size, j:j+block_size] = idct2(dct_image[i:i+block_size, j:j+block_size])
    return image


# ─────────────────────────────────────────────────────────────────────────────
# 3.  QIM — INSERTION
# ─────────────────────────────────────────────────────────────────────────────

def qim_encode(value: float, bit: int, delta: float) -> float:
    """
    QIM : quantifie `value` sur la grille correspondant au bit `bit`.

    Pour bit=0 : grilles paires  → arrondi sur 0, ±2Δ, ±4Δ…
    Pour bit=1 : grilles impaires → arrondi sur ±Δ, ±3Δ…
    """
    if bit == 0:
        return delta * np.round(value / delta)
    else:
        return delta * (np.round((value - delta / 2) / delta) + 0.5)


def embed_watermark(image: np.ndarray,
                    watermark: np.ndarray,
                    delta: float = 25.0,
                    block_size: int = 8,
                    secret_key: int = 1234) -> np.ndarray:
    """
    Insère le watermark dans l'image via QIM-DCT.

    Paramètres
    ----------
    image      : image hôte en niveaux de gris (float64)
    watermark  : tableau binaire 1D
    delta      : pas de quantification (contrôle robustesse/invisibilité)
    block_size : taille des blocs DCT
    secret_key : graine pour la sélection des coefficients

    Retourne l'image tatouée (float64).
    """
    h, w = image.shape
    n_bits = len(watermark)

    # DCT par blocs
    dct_img = apply_dct_blocks(image, block_size)

    # Lister tous les blocs disponibles
    blocks = [(i, j)
              for i in range(0, h - block_size + 1, block_size)
              for j in range(0, w - block_size + 1, block_size)]

    if n_bits > len(blocks):
        raise ValueError(f"Watermark trop long ({n_bits} bits) pour l'image ({len(blocks)} blocs).")

    # Sélection des coefficients de moyenne fréquence
    coeff_positions = select_coefficients(1, block_size, secret_key)
    coeff_r, coeff_c = coeff_positions[0]  # Un coefficient par bloc

    # Insertion bit par bit
    random.seed(secret_key)
    chosen_blocks = random.sample(blocks, n_bits)

    for idx, (br, bc) in enumerate(chosen_blocks):
        coeff = dct_img[br + coeff_r, bc + coeff_c]
        dct_img[br + coeff_r, bc + coeff_c] = qim_encode(coeff, watermark[idx], delta)

    # DCT inverse
    watermarked = apply_idct_blocks(dct_img, block_size)
    return np.clip(watermarked, 0, 255)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  QIM — EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def qim_decode(value: float, delta: float) -> int:
    """
    Décode un bit depuis un coefficient QIM.
    Compare la distance au quantificateur pair vs impair.
    """
    q0 = delta * np.round(value / delta)
    q1 = delta * (np.round((value - delta / 2) / delta) + 0.5)
    return 0 if abs(value - q0) < abs(value - q1) else 1


def extract_watermark(watermarked: np.ndarray,
                      n_bits: int,
                      delta: float = 25.0,
                      block_size: int = 8,
                      secret_key: int = 1234) -> np.ndarray:
    """
    Extrait le watermark de l'image tatouée (ou attaquée).
    """
    h, w = watermarked.shape
    dct_img = apply_dct_blocks(watermarked, block_size)

    blocks = [(i, j)
              for i in range(0, h - block_size + 1, block_size)
              for j in range(0, w - block_size + 1, block_size)]

    coeff_positions = select_coefficients(1, block_size, secret_key)
    coeff_r, coeff_c = coeff_positions[0]

    random.seed(secret_key)
    chosen_blocks = random.sample(blocks, n_bits)

    extracted = np.zeros(n_bits, dtype=np.int32)
    for idx, (br, bc) in enumerate(chosen_blocks):
        coeff = dct_img[br + coeff_r, bc + coeff_c]
        extracted[idx] = qim_decode(coeff, delta)

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calcule le PSNR entre deux images (dB). Plus élevé = meilleure qualité."""
    orig_u8 = np.clip(original, 0, 255).astype(np.uint8)
    proc_u8 = np.clip(processed, 0, 255).astype(np.uint8)
    return psnr(orig_u8, proc_u8, data_range=255)


def compute_ber(original_wm: np.ndarray, extracted_wm: np.ndarray) -> float:
    """
    Calcule le Bit Error Rate (BER).
    BER = 0.0 → extraction parfaite.
    BER = 0.5 → extraction aléatoire.
    """
    return np.mean(original_wm != extracted_wm)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ATTAQUES
# ─────────────────────────────────────────────────────────────────────────────

def attack_gaussian_noise(image: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Attaque par bruit gaussien additif."""
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255)


def attack_jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """Attaque par compression JPEG (simulée via encode/decode OpenCV)."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img_u8, encode_param)
    decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return decoded.astype(np.float64)


def attack_crop(image: np.ndarray, crop_ratio: float = 0.1) -> np.ndarray:
    """Attaque par recadrage partiel (remplit les bords avec 0)."""
    h, w = image.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    attacked = image.copy()
    attacked[:ch, :] = 0
    attacked[-ch:, :] = 0
    attacked[:, :cw] = 0
    attacked[:, -cw:] = 0
    return attacked


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_path: str | None = None,
                 watermark_size: int = 64,
                 delta: float = 25.0,
                 secret_key: int = 1234,
                 wm_seed: int = 42,
                 output_dir: str = ".") -> dict:
    """
    Exécute le pipeline complet de tatouage QIM.

    Paramètres
    ----------
    image_path    : chemin vers l'image hôte (crée une image synthétique si None)
    watermark_size: nombre de bits du watermark
    delta         : pas de quantification QIM
    secret_key    : clé secrète pour la sélection des coefficients
    wm_seed       : graine pour la génération du watermark
    output_dir    : répertoire de sortie

    Retourne un dict avec les métriques de chaque scénario.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Chargement / création de l'image ──────────────────────────────────────
    if image_path and os.path.exists(image_path):
        host = load_image_gray(image_path)
        # Recadrer à un multiple de 8
        h8 = (host.shape[0] // 8) * 8
        w8 = (host.shape[1] // 8) * 8
        host = host[:h8, :w8]
        print(f"[INFO] Image chargée : {image_path}  ({host.shape[0]}×{host.shape[1]})")
    else:
        print("[INFO] Aucune image fournie — génération d'une image synthétique (512×512).")
        # Image synthétique : dégradé + bruit structuré
        x = np.linspace(0, 255, 512)
        host = np.tile(x, (512, 1))
        host += np.random.default_rng(0).normal(0, 5, host.shape)
        host = np.clip(host, 0, 255)

    # ── Watermark ─────────────────────────────────────────────────────────────
    watermark = generate_watermark(watermark_size, seed=wm_seed)
    print(f"[INFO] Watermark ({watermark_size} bits) : {watermark[:16]}…")

    # ── Insertion ─────────────────────────────────────────────────────────────
    watermarked = embed_watermark(host, watermark, delta=delta, secret_key=secret_key)
    psnr_embed = compute_psnr(host, watermarked)
    print(f"\n{'─'*50}")
    print(f"  PSNR après insertion  : {psnr_embed:.2f} dB")

    # ── Extraction sans attaque ────────────────────────────────────────────────
    ext_clean = extract_watermark(watermarked, watermark_size, delta=delta, secret_key=secret_key)
    ber_clean = compute_ber(watermark, ext_clean)
    print(f"  BER sans attaque      : {ber_clean:.4f}  ({int((1-ber_clean)*100)}% bits corrects)")

    # ── Scénarios d'attaques ───────────────────────────────────────────────────
    results = {
        "Sans attaque":      {"psnr": psnr_embed, "ber": ber_clean, "image": watermarked.copy()},
    }

    attacks = {
        "Bruit gaussien σ=5":  lambda img: attack_gaussian_noise(img, sigma=5),
        "Bruit gaussien σ=15": lambda img: attack_gaussian_noise(img, sigma=15),
        "JPEG Q=80":           lambda img: attack_jpeg_compression(img, quality=80),
        "JPEG Q=50":           lambda img: attack_jpeg_compression(img, quality=50),
        "JPEG Q=20":           lambda img: attack_jpeg_compression(img, quality=20),
        "Recadrage 10%":       lambda img: attack_crop(img, 0.10),
    }

    for name, attack_fn in attacks.items():
        attacked = attack_fn(watermarked)
        ext = extract_watermark(attacked, watermark_size, delta=delta, secret_key=secret_key)
        ber = compute_ber(watermark, ext)
        p = compute_psnr(watermarked, attacked)
        results[name] = {"psnr": p, "ber": ber, "image": attacked}
        print(f"  [{name:22s}]  PSNR={p:6.2f} dB   BER={ber:.4f}")

    print(f"{'─'*50}\n")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_results(host, watermarked, watermark, results, output_dir)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _plot_results(host, watermarked, watermark, results, output_dir):
    """Génère et sauvegarde les figures de résultats."""

    # ── Figure 1 : Original vs Tatoué + différence ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Tatouage QIM-DCT : Original vs Tatoué", fontsize=14, fontweight='bold')

    axes[0].imshow(host, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Image hôte (originale)")
    axes[0].axis('off')

    axes[1].imshow(watermarked, cmap='gray', vmin=0, vmax=255)
    psnr_val = results["Sans attaque"]["psnr"]
    axes[1].set_title(f"Image tatouée\nPSNR = {psnr_val:.2f} dB")
    axes[1].axis('off')

    diff = np.abs(watermarked - host)
    im = axes[2].imshow(diff * 10, cmap='hot', vmin=0, vmax=255)
    axes[2].set_title("Différence × 10\n(perturbation QIM)")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p1 = os.path.join(output_dir, "fig1_original_vs_tatouee.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {p1}")

    # ── Figure 2 : Watermarks (original vs extrait) ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    fig.suptitle("Watermark : original vs extrait", fontsize=13, fontweight='bold')

    wm_2d_len = int(np.ceil(np.sqrt(len(watermark))))
    wm_padded = np.zeros(wm_2d_len * wm_2d_len)
    wm_padded[:len(watermark)] = watermark
    wm_img = wm_padded.reshape(wm_2d_len, wm_2d_len)

    axes[0].imshow(wm_img, cmap='gray', interpolation='nearest')
    axes[0].set_title("Watermark original")
    axes[0].axis('off')

    ext_clean = results["Sans attaque"]
    ext_arr = extract_watermark(results["Sans attaque"]["image"],
                                 len(watermark), delta=25.0)
    ext_padded = np.zeros(wm_2d_len * wm_2d_len)
    ext_padded[:len(watermark)] = ext_arr
    axes[1].imshow(ext_padded.reshape(wm_2d_len, wm_2d_len), cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Extrait (sans attaque)\nBER = {results['Sans attaque']['ber']:.4f}")
    axes[1].axis('off')

    # Extrait après JPEG Q=50
    key = "JPEG Q=50"
    ext_arr2 = extract_watermark(results[key]["image"], len(watermark), delta=25.0)
    ext_padded2 = np.zeros(wm_2d_len * wm_2d_len)
    ext_padded2[:len(watermark)] = ext_arr2
    axes[2].imshow(ext_padded2.reshape(wm_2d_len, wm_2d_len), cmap='gray', interpolation='nearest')
    axes[2].set_title(f"Extrait après {key}\nBER = {results[key]['ber']:.4f}")
    axes[2].axis('off')

    plt.tight_layout()
    p2 = os.path.join(output_dir, "fig2_watermarks.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {p2}")

    # ── Figure 3 : Tableau récapitulatif des métriques ────────────────────────
    scenarios = list(results.keys())
    bers  = [results[s]["ber"]  for s in scenarios]
    psnrs = [results[s]["psnr"] for s in scenarios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Robustesse du système QIM face aux attaques", fontsize=13, fontweight='bold')

    colors_ber  = ['#2ecc71' if b < 0.05 else '#e74c3c' if b > 0.2 else '#f39c12' for b in bers]
    bars1 = ax1.bar(scenarios, bers, color=colors_ber, edgecolor='white', linewidth=0.8)
    ax1.axhline(0.05, color='gray', linestyle='--', linewidth=1, label='Seuil 5%')
    ax1.set_title("BER (Bit Error Rate) — plus bas = meilleur")
    ax1.set_ylabel("BER")
    ax1.set_ylim(0, max(bers) * 1.3 + 0.01)
    ax1.tick_params(axis='x', rotation=30)
    ax1.legend()
    for bar, val in zip(bars1, bers):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    colors_psnr = ['#3498db' if p > 35 else '#e74c3c' if p < 25 else '#f39c12' for p in psnrs]
    bars2 = ax2.bar(scenarios, psnrs, color=colors_psnr, edgecolor='white', linewidth=0.8)
    ax2.axhline(35, color='gray', linestyle='--', linewidth=1, label='Seuil bonne qualité (35 dB)')
    ax2.set_title("PSNR (dB) — plus haut = meilleure qualité")
    ax2.set_ylabel("PSNR (dB)")
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend()
    for bar, val in zip(bars2, psnrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    p3 = os.path.join(output_dir, "fig3_metriques.png")
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {p3}")

    # ── Figure 4 : Galerie des images attaquées ───────────────────────────────
    attack_keys = [k for k in scenarios if k != "Sans attaque"]
    n = len(attack_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Images après attaques", fontsize=13, fontweight='bold')

    for idx, key in enumerate(attack_keys):
        r, c = divmod(idx, cols)
        axes[r][c].imshow(results[key]["image"], cmap='gray', vmin=0, vmax=255)
        axes[r][c].set_title(f"{key}\nBER={results[key]['ber']:.3f} | PSNR={results[key]['psnr']:.1f} dB",
                              fontsize=10)
        axes[r][c].axis('off')

    for idx in range(len(attack_keys), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    plt.tight_layout()
    p4 = os.path.join(output_dir, "fig4_attaques.png")
    plt.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {p4}")

    # ── Figure 5 : Impact du delta (robustesse vs qualité) ────────────────────
    _plot_delta_tradeoff(watermarked.copy(), watermark, output_dir)


def _plot_delta_tradeoff(watermarked_ref, watermark, output_dir):
    """Courbe PSNR et BER en fonction du pas de quantification delta."""
    # On re-embed pour chaque delta depuis une image hôte reconstruite
    # (ici on montre juste l'impact de l'extraction sur image JPEG Q=50)
    deltas = [5, 10, 15, 20, 25, 30, 40, 50, 70, 100]

    # Créer une image hôte propre (synthétique)
    x = np.linspace(0, 255, 512)
    host_test = np.tile(x, (512, 1))

    psnrs, bers = [], []
    for d in deltas:
        wm_img = embed_watermark(host_test, watermark, delta=d, secret_key=1234)
        attacked = attack_jpeg_compression(wm_img, quality=50)
        ext = extract_watermark(attacked, len(watermark), delta=d, secret_key=1234)
        psnrs.append(compute_psnr(host_test, wm_img))
        bers.append(compute_ber(watermark, ext))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(deltas, psnrs, 'b-o', label='PSNR (dB)', linewidth=2)
    ax2.plot(deltas, bers,  'r-s', label='BER (JPEG Q=50)', linewidth=2)
    ax1.set_xlabel("Pas de quantification Δ", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", color='b', fontsize=11)
    ax2.set_ylabel("BER", color='r', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title("Compromis invisibilité / robustesse selon Δ", fontsize=13, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    p5 = os.path.join(output_dir, "fig5_delta_tradeoff.png")
    plt.savefig(p5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {p5}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mini-Projet : Tatouage numérique QIM-DCT")
    parser.add_argument("--image",    type=str,   default=None,
                        help="Chemin vers l'image hôte (PNG/JPG)")
    parser.add_argument("--bits",     type=int,   default=64,
                        help="Taille du watermark en bits (défaut: 64)")
    parser.add_argument("--delta",    type=float, default=25.0,
                        help="Pas de quantification QIM (défaut: 25.0)")
    parser.add_argument("--key",      type=int,   default=1234,
                        help="Clé secrète (défaut: 1234)")
    parser.add_argument("--output",   type=str,   default="output_watermark",
                        help="Répertoire de sortie (défaut: output_watermark)")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        watermark_size=args.bits,
        delta=args.delta,
        secret_key=args.key,
        output_dir=args.output,
    )
