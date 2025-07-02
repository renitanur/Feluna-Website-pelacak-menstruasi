import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def calculate_angle(landmark1, landmark2, landmark3):
    """Menghitung sudut antara tiga landmark."""
    x1, y1 = landmark1[0], landmark1[1]
    x2, y2 = landmark2[0], landmark2[1]
    x3, y3 = landmark3[0], landmark3[1]

    radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

def correct_feedback(model, video=0, input_csv='static/dataset/deteksi/angle_extraction.csv'):
    """Memberikan feedback pose secara real-time."""
    # Buka video atau kamera
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError("Video atau kamera tidak dapat diakses.")

    # Muat data referensi sudut dari CSV
    try:
        reference_angles = pd.read_csv(input_csv).mean(axis=0)[:12].values
    except Exception as e:
        raise ValueError(f"Gagal memuat file CSV: {e}")

    correction_tolerance = 30  # Toleransi nilai sudut
    angle_names = [
        "L-wrist", "R-wrist", "L-elbow", "R-elbow",
        "L-shoulder", "R-shoulder", "L-knee", "R-knee",
        "L-ankle", "R-ankle", "L-hip", "R-hip"
    ]

    fps_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame video.")
            break

        # Konversi frame ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Ekstraksi koordinat landmark
            landmark_coords = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in landmarks]

            # Hitung sudut setiap pasangan landmark
            angles = []
            angle_pairs = [
                [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.LEFT_INDEX.value],
                [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.RIGHT_INDEX.value],
                [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
                [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
                [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value],
                [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
                [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value],
                [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value],
                [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
                [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ]

            for pair in angle_pairs:
                angle = calculate_angle(
                    landmark_coords[pair[0]],
                    landmark_coords[pair[1]],
                    landmark_coords[pair[2]]
                )
                angles.append(angle)

            # Prediksi pose dan evaluasi
            prediction = model.predict([angles])[0]
            feedback = "CORRECT" if all(abs(a - r) <= correction_tolerance for a, r in zip(angles, reference_angles)) else "WRONG"

            # Tampilkan feedback di frame
            cv2.putText(frame, f"Pose: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Feedback: {feedback}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if feedback == "CORRECT" else (0, 0, 255), 2)

            # Gambar landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Hitung FPS
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Tampilkan frame
        cv2.imshow('Pose Feedback', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()