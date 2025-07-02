import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import *
from demo import *
import mediapipe as mp

def run_webcam_pose_detection(model, extract_pose_angles, camera_index=0, width=640, height=480, fps=30):
    """
    Membuka webcam dan menjalankan deteksi pose secara real-time.

    Parameters:
    - model: Model yang telah dilatih untuk prediksi pose.
    - extract_pose_angles: Fungsi untuk mengekstrak fitur sudut dari hasil deteksi pose.
    - camera_index: Indeks kamera (default=0).
    - width: Lebar frame kamera (default=640).
    - height: Tinggi frame kamera (default=480).
    - fps: Frames per second kamera (default=30).
    """
    # Inisialisasi MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Buka webcam dengan backend DirectShow untuk Windows
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Kamera tidak terdeteksi!")
        print("Mencoba membuka dengan pengaturan default...")
        cap = cv2.VideoCapture(camera_index)  # Coba tanpa menentukan backend
        if not cap.isOpened():
            print("Error: Masih tidak bisa membuka kamera!")
            return  # Keluar dari fungsi jika kamera masih tidak bisa dibuka

    # Atur resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    print("Kamera terbuka dengan sukses. Tekan 'q' atau 'esc' untuk keluar.")

    try:
        while True:
            ret, frame = cap.read()  # Tangkap frame dari webcam
            if not ret:
                print("Error: Tidak bisa membaca dari webcam!")
                break

            # Konversi frame ke RGB untuk pemrosesan MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Ekstrak fitur sudut dari landmarks
                features = extract_pose_angles(results)
                
                # Pastikan format fitur yang benar
                features = np.array(features).reshape(1, -1)  # Sesuaikan format input ke model
                
                # Prediksi pose menggunakan model yang telah dilatih
                try:
                    prediction = model.predict(features)[0]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    prediction = "Unknown"

                # Gambar landmarks dan prediksi pada frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"Pose: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Tampilkan frame
            cv2.imshow("Pose Detection", frame)

            # Keluar jika menekan 'q' atau 'esc'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 adalah kode untuk tombol Escape
                print("Keluar dari aplikasi...")
                break
    except Exception as e:
        print(f"Error during webcam operation: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Load data
    data_train = pd.read_csv(os.path.join('static', 'dataset', 'deteksi', 'training_data.csv'))
    data_test = pd.read_csv(os.path.join('static', 'dataset', 'deteksi', 'test_data.csv'))

    X, Y = data_train.iloc[:, :-1], data_train['target']

    # Load atau latih model
    try:
        model = load(os.path.join('model', 'deteksi', 'pose_model.joblib'))
        print("Model loaded successfully.")
    except FileNotFoundError:
        model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
        model.fit(X, Y)
        dump(model, os.path.join('model', 'deteksi', 'pose_model.joblib'))
        print("Model trained and saved successfully.")

    # Evaluasi model
    predictions = evaluate(data_test, model, show=True)

    # Plot confusion matrix
    cm = confusion_matrix(data_test['target'], predictions)
    plt.figure(figsize=(11, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Buka webcam dan lakukan prediksi real-time
    run_webcam_pose_detection(model, extract_pose_angles, camera_index=0, width=640, height=480, fps=30)

if __name__ == "__main__":
    main()
