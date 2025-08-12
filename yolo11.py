from ultralytics import YOLO
import cv2
import json
import numpy as np
import datetime
import os

# YOLO11 포즈 모델 로드
model = YOLO('yolo11n-pose.pt')

# 웹캠 연결
cap = cv2.VideoCapture(0)

# 저장할 데이터 리스트
pose_data = []
frame_count = 0

# 저장 폴더 생성
save_dir = "pose_outputs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 타임스탬프 생성
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print("포즈 데이터 수집 시작!")
print("'q': 종료 및 저장")
print("'s': 현재 프레임 저장")
print("'r': 비디오 녹화 시작/정지")

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 포즈 추정 실행
    results = model(frame)
    
    # 결과 데이터 추출
    if results[0].keypoints is not None:
        # 키포인트 데이터 추출
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2) - N명의 사람
        keypoints_conf = results[0].keypoints.conf.cpu().numpy()  # (N, 17) - 신뢰도
        
        # 바운딩 박스 추출
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4) - x1,y1,x2,y2
            box_conf = results[0].boxes.conf.cpu().numpy()  # (N,) - 탐지 신뢰도
        else:
            boxes = None
            box_conf = None
        
        # 프레임별 데이터 저장
        frame_data = {
            'frame_number': frame_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'num_people': len(keypoints_xy),
            'people': []
        }
        
        # 각 사람별 데이터 저장
        for person_idx in range(len(keypoints_xy)):
            person_data = {
                'person_id': person_idx,
                'keypoints': {
                    'coordinates': keypoints_xy[person_idx].tolist(),  # [[x,y], [x,y], ...]
                    'confidence': keypoints_conf[person_idx].tolist(),  # [conf1, conf2, ...]
                    'visibility': (keypoints_conf[person_idx] > 0.5).tolist()  # 보이는지 여부
                },
                'bounding_box': {
                    'coordinates': boxes[person_idx].tolist() if boxes is not None else None,
                    'confidence': float(box_conf[person_idx]) if box_conf is not None else None
                }
            }
            
            # 관절별 상세 정보 추가
            keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            person_data['keypoints']['named_points'] = {}
            for i, name in enumerate(keypoint_names):
                person_data['keypoints']['named_points'][name] = {
                    'x': float(keypoints_xy[person_idx][i][0]),
                    'y': float(keypoints_xy[person_idx][i][1]),
                    'confidence': float(keypoints_conf[person_idx][i]),
                    'visible': bool(keypoints_conf[person_idx][i] > 0.5)
                }
            
            frame_data['people'].append(person_data)
        
        pose_data.append(frame_data)
    
    # 결과를 프레임에 그리기
    annotated_frame = results[0].plot()
    
    # 정보 텍스트 추가
    cv2.putText(annotated_frame, f'Frame: {frame_count}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if results[0].keypoints is not None:
        cv2.putText(annotated_frame, f'People: {len(keypoints_xy)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 녹화 상태 표시
    if recording:
        cv2.putText(annotated_frame, 'RECORDING', (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 화면에 출력
    cv2.imshow('YOLO11 Pose', annotated_frame)
    
    # 비디오 녹화
    if recording and video_writer is not None:
        video_writer.write(annotated_frame)
    
    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 현재 프레임 이미지 저장
        cv2.imwrite(f'{save_dir}/frame_{frame_count}_{timestamp}.jpg', annotated_frame)
        print(f"프레임 {frame_count} 저장됨")
    elif key == ord('r'):
        # 비디오 녹화 토글
        if not recording:
            video_path = f'{save_dir}/pose_video_{timestamp}.avi'
            height, width = annotated_frame.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
            recording = True
            print("비디오 녹화 시작")
        else:
            if video_writer is not None:
                video_writer.release()
            recording = False
            print("비디오 녹화 종료")
    
    frame_count += 1

# 정리
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

# JSON 파일로 포즈 데이터 저장
json_path = f'{save_dir}/pose_data_{timestamp}.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(pose_data, f, indent=2, ensure_ascii=False)

# CSV 파일로도 저장 (간단한 형태)
import pandas as pd

if pose_data:
    # 첫 번째 사람의 데이터만 CSV로 저장
    csv_data = []
    for frame in pose_data:
        if frame['people']:
            person = frame['people'][0]  # 첫 번째 사람
            row = {
                'frame': frame['frame_number'],
                'timestamp': frame['timestamp']
            }
            # 각 관절점의 x, y, confidence 추가
            for name, point in person['keypoints']['named_points'].items():
                row[f'{name}_x'] = point['x']
                row[f'{name}_y'] = point['y']
                row[f'{name}_conf'] = point['confidence']
            csv_data.append(row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = f'{save_dir}/pose_data_{timestamp}.csv'
        df.to_csv(csv_path, index=False)

print(f"\n=== 저장 완료 ===")
print(f"총 {frame_count}개 프레임 처리")
print(f"포즈 데이터: {len(pose_data)}개 프레임")
print(f"JSON 파일: {json_path}")
if 'csv_path' in locals():
    print(f"CSV 파일: {csv_path}")
print(f"저장 폴더: {save_dir}/")

# 저장된 데이터 샘플 출력
if pose_data:
    print(f"\n=== 데이터 샘플 ===")
    sample = pose_data[0] if pose_data else {}
    print(f"첫 번째 프레임 데이터:")
    print(f"- 탐지된 사람 수: {sample.get('num_people', 0)}")
    if sample.get('people'):
        person = sample['people'][0]
        print(f"- 첫 번째 사람의 코 위치: {person['keypoints']['named_points']['nose']}")
        print(f"- 어깨 위치: 왼쪽{person['keypoints']['named_points']['left_shoulder']}")
        print(f"            오른쪽{person['keypoints']['named_points']['right_shoulder']}")