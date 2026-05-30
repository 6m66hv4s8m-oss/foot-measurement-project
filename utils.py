import numpy as np
import os
import gdown
from ultralytics import YOLO

# 💡 보정 계수 및 설정
LENGTH_CORRECTION_FACTOR = 0.65 
WIDTH_CORRECTION_FACTOR = 0.55   
REAL_CARD_WIDTH_MM = 85.6

# 💡 여기에 발 모델의 구글 드라이브 파일 ID를 넣으세요!
FOOT_MODEL_ID = ""

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"모델 파일 다운로드 중: {output_path}...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    card_path = os.path.join(base_dir, "card_best.pt")
    foot_path = os.path.join(base_dir, "foot_best.pt")
    
    # 1. 발 모델은 구글 드라이브에서 다운로드
    download_model(FOOT_MODEL_ID, foot_path)
    
    # 2. 카드 모델은 GitHub에 이미 올라와 있으므로 바로 로드
    return YOLO(card_path), YOLO(foot_path)

def calculate_measurements(card_res, foot_res):
    # 카드 인식 (OBB 방식 + 신뢰도 높은 1개 선택)
    card_w = 150.0
    if card_res.obb is not None and len(card_res.obb) > 0:
        best_card = card_res.obb[card_res.obb.conf.argmax()]
        w = float(best_card.xywhr[0][2])
        h = float(best_card.xywhr[0][3])
        card_w = max(w, h)
    
    mm_per_pixel = REAL_CARD_WIDTH_MM / card_w
    
    # 발 측정 (신뢰도 높은 1개 선택 + 보정 적용)
    length, width = 0.0, 0.0
    if foot_res.keypoints is not None and len(foot_res.keypoints.xy) > 0:
        best_foot_idx = foot_res.boxes.conf.argmax()
        kpts = foot_res.keypoints.xy[best_foot_idx].cpu().numpy()
        
        # 길이: 가장 먼 거리 계산
        max_d = 0
        for i in range(len(kpts)):
            for j in range(i+1, len(kpts)):
                dist = np.linalg.norm(kpts[i]-kpts[j])
                if dist > max_d: max_d = dist
        length = (max_d * mm_per_pixel) * LENGTH_CORRECTION_FACTOR
        
        # 발볼: X축 좌표 범위 계산
        xs = [kpts[i][0] for i in range(len(kpts))]
        width = (max(xs) - min(xs)) * mm_per_pixel * WIDTH_CORRECTION_FACTOR
            
    return length, width