import streamlit as st
import cv2
import tempfile
import numpy as np
import utils

st.set_page_config(page_title="AI 발 치수 시스템", layout="centered")
st.title("👣 AI 발 치수 자동 측정 시스템")

card_model, foot_model = utils.load_models()
uploaded_file = st.file_uploader("동영상 업로드", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps / 2), 2)
    
    collected_lengths, collected_widths = [], []
    frame_count = 0
    
    with st.spinner("분석 중... 잠시만 기다려주세요!"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_interval == 0:
                card_res = card_model(frame, conf=0.1, verbose=False)[0]
                foot_res = foot_model(frame, conf=0.1, verbose=False)[0]
                
                l, w = utils.calculate_measurements(card_res, foot_res)
                
                # 유효한 값인지 체크 후 리스트 추가
                if 100 < l < 400: collected_lengths.append(l)
                if 50 < w < 200: collected_widths.append(w)
            frame_count += 1
    
    cap.release()
    
    # 결과가 있는지 확인
    f_len = np.median(collected_lengths) if collected_lengths else 0.0
    f_wid = np.median(collected_widths) if collected_widths else 0.0
    
    st.success("✅ 분석 완료!")
    
    # 💡 [핵심] 컬럼을 나누어 나란히 출력
    col1, col2 = st.columns(2)
    col1.metric("📏 최종 발 길이", f"{f_len:.1f} mm")
    col2.metric("🦶 최종 발볼 넓이", f"{f_wid:.1f} mm")