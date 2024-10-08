from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import numpy as np
import cv2
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/watermark'
app.config['ALLOWED_EXTENSIONS'] = {'jpg','png'}

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def dct2(block):
    return cv2.dct(np.float32(block))

def idct2(block):
    return cv2.idct(block)

def zigzag_scan(block):
    zigzag_order = [
        (0,0),
        (0,1),(1,0),
        (2,0),(1,1),(0,2),
        (0,3),(1,2),(2,1),(3,0),
        (4,0),(3,1),(2,2),(1,3),(0,4),
        (0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
        (6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),
        (0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0),
        (7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),
        (2,7),(3,6),(4,5),(5,4),(6,3),(7,2),
        (7,3),(6,4),(5,5),(4,6),(3,7),
        (4,7),(5,6),(6,5),(7,4),
        (7,5),(6,6),(5,7),
        (6,7),(7,6),
        (7,7)
    ]
    return [block[i][j] for i,j in zigzag_order]

def embed_watermark(original_image_path, watermark_image_path, output_path, watermark_strength=0.2):  # เพิ่มพารามิเตอร์
    # อ่านภาพต้นฉบับและลายน้ำเป็น RGB
    original = cv2.imread(original_image_path)
    watermark = cv2.imread(watermark_image_path)

    # ขนาดของภาพลายน้ำต้องเท่ากับภาพต้นฉบับ
    if original.shape[:2] != watermark.shape[:2]:
        # ปรับขนาดภาพลายน้ำให้เท่ากับภาพต้นฉบับ
        watermark = cv2.resize(watermark, (original.shape[1], original.shape[0]))

    height, width, _ = original.shape
    block_size = 8

    # แปลงภาพเป็นบล็อก 8x8
    watermarked = original.copy()

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # ดึงบล็อก 8x8 จากภาพต้นฉบับ
            original_block = original[i:i+block_size, j:j+block_size]
            watermark_block = watermark[i:i+block_size, j:j+block_size]

            # ตรวจสอบขนาดบล็อก
            if original_block.shape[0] != block_size or original_block.shape[1] != block_size:
                continue

            # ทำ DCT สำหรับแต่ละช่องสี
            dct_original_blocks = []
            dct_watermark_blocks = []

            for channel in range(3):  # RGB มี 3 ช่องสี
                dct_original = dct2(original_block[:, :, channel])
                dct_watermark = dct2(watermark_block[:, :, channel])
                dct_original_blocks.append(dct_original)
                dct_watermark_blocks.append(dct_watermark)

                # ฝังลายน้ำใน coefficients ความถี่ต่ำ (ตัวอย่างที่ [1][1]) โดยเพิ่มความเข้มของลายน้ำ
                dct_original[1][1] += watermark_strength * dct_watermark[1][1]  # ปรับค่าความเข้มที่นี่

            # ทำ inverse DCT สำหรับแต่ละช่องสี
            for channel in range(3):
                idct_block = idct2(dct_original_blocks[channel])
                idct_block = np.clip(idct_block, 0, 255)
                watermarked[i:i+block_size, j:j+block_size, channel] = idct_block

    # บันทึกภาพที่ถูกฝังลายน้ำ
    cv2.imwrite(output_path, watermarked)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'original_image' not in request.files or 'watermark_image' not in request.files:
            return redirect(request.url)

        original_file = request.files['original_image']
        watermark_file = request.files['watermark_image']
        strength = float(request.form['strength'])  # อ่านค่าความเข้มจากฟอร์ม (เพิ่มบรรทัดนี้)

        if original_file and allowed_file(original_file.filename) and watermark_file and allowed_file(watermark_file.filename):
            original_filename = secure_filename(original_file.filename)
            watermark_filename = secure_filename(watermark_file.filename)

            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)

            original_file.save(original_path)
            watermark_file.save(watermark_path)

            result_filename = 'watermarked_' + original_filename
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            # ฝังลายน้ำพร้อมกับความเข้มที่ผู้ใช้ระบุ
            embed_watermark(original_path, watermark_path, result_path, watermark_strength=strength)  # ส่งค่า strength ที่อ่านได้

            return render_template('index.html', result_image=result_filename)

    return render_template('index.html', result_image=None)

@app.route('/watermark/<filename>')
def send_watermarked_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
