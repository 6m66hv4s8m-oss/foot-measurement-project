import numpy as np
from ultralytics import YOLO
import os

# 💡 [최종 보정] 오차(10cm)를 잡기 위한 강제 다이어트 계수
LENGTH_CORRECTION_FACTOR = 0.7
WIDTH_CORRECTION_FACTOR = 0.15
REAL_CARD_WIDTH_MM = 85.6

def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_card_path = os.path.join(base_dir, "models", "card_best.pt")
    model_foot_path = os.path.join(base_dir, "models", "foot_best.pt")
    return YOLO(model_card_path), YOLO(model_foot_path)

def calculate_measurements(card_res, foot_res):
    # 카드 너비 (OBB 방식 + 신뢰도 높은 1개)
    card_w = 150.0
    if card_res.obb is not None and len(card_res.obb) > 0:
        best_card = card_res.obb[card_res.obb.conf.argmax()]
        w = float(best_card.xywhr[0][2])
        h = float(best_card.xywhr[0][3])
        card_w = max(w, h)
    
    mm_per_pixel = REAL_CARD_WIDTH_MM / card_w
    
    # 발 측정 (신뢰도 높은 1개 + 보정)
    length, width = 0.0, 0.0
    if foot_res.keypoints is not None and len(foot_res.keypoints.xy) > 0:
        best_foot_idx = foot_res.boxes.conf.argmax()
        kpts = foot_res.keypoints.xy[best_foot_idx].cpu().numpy()
        
        # 길이: 가장 먼 거리
        max_d = 0
        for i in range(len(kpts)):
            for j in range(i+1, len(kpts)):
                dist = np.linalg.norm(kpts[i]-kpts[j])
                if dist > max_d: max_d = dist
        length = (max_d * mm_per_pixel) * LENGTH_CORRECTION_FACTOR
        
        # 발볼: X축 기반
        xs = [kpts[i][0] for i in range(len(kpts))]
        width = (max(xs) - min(xs)) * mm_per_pixel * WIDTH_CORRECTION_FACTOR
            
    return length, width