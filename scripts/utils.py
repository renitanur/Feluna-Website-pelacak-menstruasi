import mediapipe as mp
import cv2
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
points = mp_pose.PoseLandmark  
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(landmark1, landmark2, landmark3):
    """Menghitung sudut antara tiga landmark."""
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    return angle

def extract_pose_angles(results):
    """Ekstrak sudut pose jika landmark tersedia."""
    angles = []
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark

            # Menghitung sudut berbagai bagian tubuh
            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]))

            angles.append(calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]))

        except IndexError:
            logging.error("Error: Landmark tidak lengkap untuk perhitungan sudut.")
    return angles

def predict(img_path, model, show=False):
    """Prediksi pose berdasarkan gambar."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Gambar {img_path} tidak ditemukan.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        angles = extract_pose_angles(results)
        if angles:
            prediction = model.predict([angles])[0]

            if show:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(img, str(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Prediction", img)
                cv2.waitKey(0)

            return prediction
        else:
            logging.error("Sudut tidak dapat diekstrak dari gambar.")
            return None
    else:
        logging.error("Pose tidak terdeteksi pada gambar.")
        return None

def predict_frame(frame, model):
    """
    Fungsi untuk memprediksi pose dari satu frame gambar menggunakan model yang diberikan.
    
    Args:
        frame (numpy.ndarray): Frame gambar yang akan diproses.
        model: Model pose detection yang telah dimuat.
    
    Returns:
        dict: Hasil deteksi pose.
    """
    try:
        logging.debug("Memprediksi pose untuk frame yang diambil")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            angles = extract_pose_angles(results)
            if angles:
                prediction = model.predict([angles])[0]
                logging.debug(f"Hasil prediksi pose: {prediction}")
                return {"pose": prediction}
            else:
                logging.error("Sudut tidak dapat diekstrak dari frame.")
                return {"pose": "Sudut tidak dapat diekstrak."}
        else:
            logging.error("Pose tidak terdeteksi pada frame.")
            return {"pose": "Pose tidak terdeteksi."}
    except Exception as e:
        logging.error(f"Error di predict_frame: {e}")
        return {"pose": "Error dalam deteksi pose."}

def predict_video(model, video=0, show=False):
    """Prediksi pose pada video atau webcam."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError("Video atau kamera tidak dapat diakses.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Tidak dapat membaca frame video.")
            break

        pose_result = predict_frame(frame, model)

        if show:
            if pose_result["pose"] != "Pose tidak terdeteksi." and pose_result["pose"] != "Error dalam deteksi pose.":
                mp_drawing.draw_landmarks(frame, pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, str(pose_result["pose"]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if show:
            cv2.imshow("Pose Prediction", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def evaluate(data_test, model, show=False):
    """Evaluasi performa model menggunakan dataset uji."""
    target = data_test["target"].values
    predictions = []

    for _, row in data_test.iterrows():
        angles = row.iloc[:-1].values.tolist()
        predictions.append(model.predict([angles])[0])

    if show:
        print(confusion_matrix(target, predictions))
        print(classification_report(target, predictions))

    return predictions
