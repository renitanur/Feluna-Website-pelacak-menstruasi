from flask import Flask, render_template, request, jsonify
from scripts.detect import run_webcam_pose_detection
from scripts.demo import correct_feedback 
from scripts.utils import predict_video  # Menggunakan impor relatif

import base64
import cv2
import joblib  
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the pose detection model
MODEL_PATH = "model/deteksi/pose_model.joblib"  # Lokasi model
try:
    pose_model = joblib.load(MODEL_PATH)
    logging.info(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    pose_model = None

# Route untuk menampilkan halaman deteksi
@app.route('/')
def deteksi_page():
    return render_template('deteksi.html')  # Memuat file deteksi.html dari folder templates

# Route untuk menampilkan data CSV
@app.route('/deteksi', methods=['GET'])
def display_csv():
    try:
        # Memproses CSV menggunakan fungsi correct_feedback
        csv_data = correct_feedback('static/dataset/deteksi/angle_extraction.csv')
    except FileNotFoundError:
        logging.error("CSV file not found")
        return jsonify({'error': 'CSV file not found'}), 404
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        return jsonify({'error': f'Failed to process CSV: {str(e)}'}), 500

    # Mengirimkan data CSV ke template deteksi.html
    return render_template('deteksi.html', data=csv_data.to_dict(orient='records'))

# Route untuk membuka webcam dan melakukan deteksi pose
@app.route('/run_webcam', methods=['GET'])
def run_webcam():
    try:
        run_webcam_pose_detection(pose_model)  # Panggil fungsi dengan model yang sudah dimuat
        return jsonify({'message': 'Webcam detection completed'}), 200
    except Exception as e:
        logging.error(f"Error running webcam pose detection: {str(e)}")

    # Kode berikut tidak akan pernah dijalankan karena sudah ada return di atas
    # Jika Anda berniat menggunakan predict_video, pastikan untuk mengintegrasikannya sebelum return
    
    try:
        result = predict_video(cv2.FILE_STORAGE_MEMORY, pose_model)  # Pastikan hanya argumen yang dibutuhkan diteruskan
        if result is None:
            return jsonify({'error': 'Pose detection failed'}), 500
    except Exception as e:
        logging.error(f'Error during pose detection: {str(e)}')
        return jsonify({'error': f'Pose detection failed: {str(e)}'}), 500

    return jsonify({'result': result})  # Kembalikan hasil deteksi pose dalam format JSON
    
    

if __name__ == "__main__":
    app.run(debug=True)