from ultralytics import YOLO
import cv2

# YOLO11 포즈 모델 로드
model = YOLO('yolo11n-pose.pt')

# 웹캠 연결
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 포즈 추정 실행
    results = model(frame)
    
    # 결과를 프레임에 그리기
    annotated_frame = results[0].plot()
    
    # 화면에 출력
    cv2.imshow('YOLO11 Pose', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()