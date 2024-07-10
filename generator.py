import os
import uuid
import cv2
import mediapipe as mp
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

current_directory = os.path.dirname(__file__)

def initialize_model():
    try:
        # Load YOLOv7 model
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=f'{current_directory}/yolov7/yolov7.pt')
        print("YOLOv7 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv7 model: {e}")
        model = None

    return model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.30)


def new_palm_masking(image_path, scale_factor=1.05, conf_threshold=0.5):

    # Initialize models with retry mechanism
    yolo_model = initialize_model()

    # If the initialization fails, try again
    if yolo_model is None:
        print("Retrying model initialization...")
        yolo_model = initialize_model()

    if yolo_model is None:
        print("Failed to initialize model after retrying.")
    else:
        print("Model initialized successfully.")
        
        
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image = image_rgb.copy()
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    def detect_hands(image_rgb, x_offset=0, y_offset=0):
        results = hands.process(image_rgb)
        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w = image_rgb.shape[:2]
                points = [(int(landmark.x * w) + x_offset, int(landmark.y * h) + y_offset) for landmark in hand_landmarks.landmark]
                points = np.array(points, dtype=np.int32)
                if len(points) >= 5:
                    hand_landmarks_list.append(points)
                    # Draw landmarks on the annotated image
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        cv2.line(annotated_image,
                                 (points[start_idx][0], points[start_idx][1]),
                                 (points[end_idx][0], points[end_idx][1]),
                                 (0, 255, 0), 2)
                    for point in points:
                        cv2.circle(annotated_image, (point[0], point[1]), 5, (0, 0, 255), -1)
        return hand_landmarks_list

    def create_mask(hand_landmarks_list):
        for points in hand_landmarks_list:
            ellipse = cv2.fitEllipse(points)
            center, axes, angle = ellipse
            axes = (int(axes[0] * scale_factor), int(axes[1] * scale_factor))
            cv2.ellipse(mask, (center, axes, angle), 255, -1)

    # First attempt to detect hands
    print("First attempt to detect hands")
    hand_landmarks_list = detect_hands(image_rgb)

    if len(hand_landmarks_list) < 2:
        # Use YOLOv7 to detect human and crop image if fewer than 2 hands are detected
        print("Second attempt to detect hands")
        image_rgb_copy = image_rgb.copy()
        results = yolo_model(image_rgb)
        detections = results.xyxy[0].cpu().numpy()
        max_conf = 0
        best_box = None

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0 and conf > conf_threshold and conf > max_conf:  # Check if it's a person and above confidence threshold
                max_conf = conf
                best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cropped_image = image_rgb_copy[y1:y2, x1:x2]
            hand_landmarks_list = detect_hands(cropped_image, x_offset=x1, y_offset=y1)

    if not hand_landmarks_list:
        raise Exception("Hands not visible")

    create_mask(hand_landmarks_list)

    output_path = f'{current_directory}/uploads/{str(uuid.uuid4())}.png'
    cv2.imwrite(output_path, mask)

    return output_path
