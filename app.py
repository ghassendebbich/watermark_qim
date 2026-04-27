from flask import Flask, request, send_file, render_template_string, jsonify
import numpy as np
import io
import base64
import os
from watermark_qim import (
    generate_watermark, embed_watermark, extract_watermark,
    compute_ber, compute_psnr, attack_gaussian_noise,
    attack_jpeg_compression, attack_crop
)
from PIL import Image

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Watermark QIM App</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f6f9; color: #333; }
        header { background: #2c3e50; color: white; padding: 20px 40px; }
        header h1 { font-size: 24px; }
        header p { font-size: 13px; opacity: 0.7; margin-top: 4px; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; padding: 30px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        h2 { font-size: 18px; margin-bottom: 16px; color: #2c3e50; }
        label { display: block; font-size: 13px; margin-bottom: 6px; color: #555; }
        input[type=file], input[type=number] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }
        input[type=number] { width: 200px; }
        button { background: #2c3e50; color: white; border: none; padding: 12px 28px; border-radius: 6px; font-size: 15px; cursor: pointer; }
        button:hover { background: #3d5166; }
        .result { margin-top: 20px; }
        .metrics { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
        .metric-box { background: #f0f4f8; border-radius: 8px; padding: 14px 20px; text-align: center; min-width: 120px; }
        .metric-box .value { font-size: 22px; font-weight: bold; color: #2c3e50; }
        .metric-box .label { font-size: 12px; color: #777; margin-top: 4px; }
        .images { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }
        .img-box { text-align: center; }
        .img-box img { max-width: 280px; border-radius: 6px; border: 1px solid #ddd; }
        .img-box p { font-size: 12px; margin-top: 6px; color: #555; }
        .download-btn { display: inline-block; margin-top: 10px; background: #27ae60; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; }
        .error { color: #e74c3c; background: #fdecea; padding: 12px; border-radius: 6px; margin-top: 12px; }
        .loading { display: none; color: #777; margin-top: 12px; }
        select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }
    </style>
</head>
<body>
<header>
    <h1>QIM Digital Watermarking</h1>
    <p>Embed and extract invisible watermarks using DCT-QIM</p>
</header>
<div class="container">

    <div class="card">
        <h2>Embed Watermark</h2>
        <form id="embedForm">
            <label>Host Image (PNG/JPG)</label>
            <input type="file" name="image" accept="image/*" required>
            <label>Watermark Size (bits)</label>
            <input type="number" name="bits" value="64" min="8" max="512">
            <label>Delta (quantization step)</label>
            <input type="number" name="delta" value="25" min="1" max="200" step="1">
            <label>Secret Key</label>
            <input type="number" name="key" value="1234">
            <label>Attack simulation</label>
            <select name="attack">
                <option value="none">No attack</option>
                <option value="gaussian">Gaussian noise (σ=5)</option>
                <option value="jpeg">JPEG compression (Q=80)</option>
                <option value="crop">Crop (5%)</option>
            </select>
            <button type="submit">Embed Watermark</button>
        </form>
        <div class="loading" id="loading">Processing...</div>
        <div class="result" id="result"></div>
    </div>

</div>
<script>
document.getElementById('embedForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').innerHTML = '';
    const formData = new FormData(e.target);
    try {
        const res = await fetch('/embed', { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('loading').style.display = 'none';
        if (data.error) {
            document.getElementById('result').innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }
        document.getElementById('result').innerHTML = `
            <div class="metrics">
                <div class="metric-box"><div class="value">${data.psnr} dB</div><div class="label">PSNR</div></div>
                <div class="metric-box"><div class="value">${data.ber}</div><div class="label">BER</div></div>
                <div class="metric-box"><div class="value">${data.bits} bits</div><div class="label">Watermark size</div></div>
            </div>
            <div class="images">
                <div class="img-box">
                    <img src="data:image/png;base64,${data.original}" />
                    <p>Original image</p>
                </div>
                <div class="img-box">
                    <img src="data:image/png;base64,${data.watermarked}" />
                    <p>Watermarked image</p>
                    <a class="download-btn" href="data:image/png;base64,${data.watermarked}" download="watermarked.png">Download</a>
                </div>
                ${data.attacked ? `
                <div class="img-box">
                    <img src="data:image/png;base64,${data.attacked}" />
                    <p>After attack (BER=${data.ber})</p>
                </div>` : ''}
            </div>
        `;
    } catch (err) {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('result').innerHTML = `<div class="error">Error: ${err.message}</div>`;
    }
});
</script>
</body>
</html>
'''

def img_to_base64(arr):
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def pil_to_gray_array(pil_img):
    gray = pil_img.convert('L')
    return np.array(gray, dtype=np.float64)

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/embed', methods=['POST'])
def embed():
    try:
        file = request.files.get('image')
        bits = int(request.form.get('bits', 64))
        delta = float(request.form.get('delta', 25.0))
        key = int(request.form.get('key', 1234))
        attack = request.form.get('attack', 'none')

        if not file:
            return jsonify({'error': 'No image provided'})

        pil_img = Image.open(file.stream)
        host = pil_to_gray_array(pil_img)

        watermark = generate_watermark(bits, seed=key)
        watermarked = embed_watermark(host, watermark, delta=delta, secret_key=key)
        psnr_val = round(compute_psnr(host, watermarked), 2)

        attacked_arr = None
        ber_val = 0.0

        if attack == 'gaussian':
            attacked_arr = attack_gaussian_noise(watermarked, sigma=5)
        elif attack == 'jpeg':
            attacked_arr = attack_jpeg_compression(watermarked, quality=80)
        elif attack == 'crop':
            attacked_arr = attack_crop(watermarked, crop_ratio=0.05)

        if attacked_arr is not None:
            extracted = extract_watermark(attacked_arr, bits, delta=delta, secret_key=key)
            ber_val = round(compute_ber(watermark, extracted), 4)
        else:
            extracted = extract_watermark(watermarked, bits, delta=delta, secret_key=key)
            ber_val = round(compute_ber(watermark, extracted), 4)

        result = {
            'psnr': psnr_val,
            'ber': ber_val,
            'bits': bits,
            'original': img_to_base64(host),
            'watermarked': img_to_base64(watermarked),
        }
        if attacked_arr is not None:
            result['attacked'] = img_to_base64(attacked_arr)

        return jsonify(result)

    except Exception as ex:
        return jsonify({'error': str(ex)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
