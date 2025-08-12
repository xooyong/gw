from ultralytics import YOLO
import cv2
import json
import numpy as np
import datetime
import os
from pathlib import Path

class VideoPoseAnalyzer:
    def __init__(self, model_path='yolo11n-pose.pt'):
        # YOLO11 포즈 모델 로드
        self.model = YOLO(model_path)
        
        # 저장 폴더 생성
        self.save_dir = "video_pose_outputs"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 관절 이름 정의
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def analyze_video(self, video_path, save_output=True, show_video=True):
        """
        비디오 파일에서 포즈 분석
        
        Args:
            video_path: 입력 비디오 파일 경로
            save_output: 결과 저장 여부
            show_video: 실시간 화면 표시 여부
        """
        # 비디오 파일 체크
        if not os.path.exists(video_path):
            print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
            return None
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
            return None
        
        # 비디오 정보 추출
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📹 비디오 정보:")
        print(f"   - 파일: {Path(video_path).name}")
        print(f"   - 해상도: {width}x{height}")
        print(f"   - FPS: {fps}")
        print(f"   - 총 프레임: {total_frames}")
        print(f"   - 재생시간: {duration:.2f}초")
        
        # 타임스탬프 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        
        # 출력 비디오 설정
        output_video_path = None
        video_writer = None
        if save_output:
            output_video_path = f'{self.save_dir}/{video_name}_pose_output_{timestamp}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 데이터 저장용 리스트
        pose_data = []
        frame_count = 0
        
        print(f"\n🚀 포즈 분석 시작...")
        print(f"   ESC: 중단, SPACE: 일시정지/재생")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✅ 비디오 분석 완료!")
                    break
                
                # 포즈 추정 실행
                results = self.model(frame, verbose=False)
                
                # 결과 데이터 추출 및 저장
                frame_data = self.extract_pose_data(results, frame_count, frame_count/fps)
                if frame_data:
                    pose_data.append(frame_data)
                
                # 결과를 프레임에 그리기
                annotated_frame = results[0].plot()
                
                # 진행률 표시
                progress = (frame_count + 1) / total_frames * 100
                time_current = frame_count / fps
                
                # 정보 텍스트 추가
                cv2.putText(annotated_frame, f'Frame: {frame_count+1}/{total_frames}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Progress: {progress:.1f}%', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Time: {time_current:.2f}s / {duration:.2f}s', 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if results[0].keypoints is not None:
                    num_people = len(results[0].keypoints.xy)
                    cv2.putText(annotated_frame, f'People: {num_people}', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 출력 비디오에 저장
                if save_output and video_writer is not None:
                    video_writer.write(annotated_frame)
                
                frame_count += 1
            else:
                # 일시정지 중일 때는 같은 프레임 표시
                annotated_frame = results[0].plot() if 'results' in locals() else frame
                cv2.putText(annotated_frame, 'PAUSED - Press SPACE to continue', 
                           (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 화면에 출력
            if show_video:
                cv2.imshow(f'YOLO11 Pose Analysis - {Path(video_path).name}', annotated_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    print("❌ 사용자가 분석을 중단했습니다.")
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print("⏸️ 일시정지" if paused else "▶️ 재생")
            else:
                # 화면 출력 없이 빠른 처리
                if frame_count % 30 == 0:  # 30프레임마다 진행률 출력
                    print(f"진행률: {progress:.1f}% ({frame_count+1}/{total_frames})")
        
        # 정리
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # 결과 저장
        if save_output and pose_data:
            self.save_results(pose_data, video_name, timestamp, video_path)
        
        return {
            'pose_data': pose_data,
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'duration': duration,
                'analyzed_frames': len(pose_data)
            },
            'output_video': output_video_path
        }
    
    def extract_pose_data(self, results, frame_number, timestamp_sec):
        """포즈 데이터 추출"""
        if results[0].keypoints is None:
            return None
        
        # 키포인트 데이터 추출
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()
        keypoints_conf = results[0].keypoints.conf.cpu().numpy()
        
        # 바운딩 박스 추출
        boxes = None
        box_conf = None
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            box_conf = results[0].boxes.conf.cpu().numpy()
        
        # 프레임 데이터 구성
        frame_data = {
            'frame_number': frame_number,
            'timestamp_sec': timestamp_sec,
            'num_people': len(keypoints_xy),
            'people': []
        }
        
        # 각 사람별 데이터 저장
        for person_idx in range(len(keypoints_xy)):
            person_data = {
                'person_id': person_idx,
                'keypoints': {
                    'coordinates': keypoints_xy[person_idx].tolist(),
                    'confidence': keypoints_conf[person_idx].tolist(),
                    'named_points': {}
                },
                'bounding_box': {
                    'coordinates': boxes[person_idx].tolist() if boxes is not None else None,
                    'confidence': float(box_conf[person_idx]) if box_conf is not None else None
                }
            }
            
            # 관절별 상세 정보
            for i, name in enumerate(self.keypoint_names):
                person_data['keypoints']['named_points'][name] = {
                    'x': float(keypoints_xy[person_idx][i][0]),
                    'y': float(keypoints_xy[person_idx][i][1]),
                    'confidence': float(keypoints_conf[person_idx][i]),
                    'visible': bool(keypoints_conf[person_idx][i] > 0.5)
                }
            
            frame_data['people'].append(person_data)
        
        return frame_data
    
    def save_results(self, pose_data, video_name, timestamp, original_video_path):
        """결과 저장"""
        # JSON 저장
        json_path = f'{self.save_dir}/{video_name}_pose_data_{timestamp}.json'
        
        # 메타데이터 추가
        output_data = {
            'metadata': {
                'original_video': original_video_path,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'total_frames_analyzed': len(pose_data),
                'model_used': 'yolo11n-pose'
            },
            'frames': pose_data
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # CSV 저장 (첫 번째 사람만)
        csv_data = []
        for frame in pose_data:
            if frame['people']:
                person = frame['people'][0]
                row = {
                    'frame': frame['frame_number'],
                    'timestamp_sec': frame['timestamp_sec']
                }
                for name, point in person['keypoints']['named_points'].items():
                    row[f'{name}_x'] = point['x']
                    row[f'{name}_y'] = point['y']
                    row[f'{name}_conf'] = point['confidence']
                csv_data.append(row)
        
        if csv_data:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_path = f'{self.save_dir}/{video_name}_pose_data_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
        
        print(f"\n💾 결과 저장 완료:")
        print(f"   JSON: {json_path}")
        if csv_data:
            print(f"   CSV:  {csv_path}")

# 사용 예시
if __name__ == "__main__":
    # 분석기 초기화
    analyzer = VideoPoseAnalyzer('yolo11n-pose.pt')
    
    # 비디오 파일 경로 (여기에 실제 파일 경로 입력)
    video_files = [
        "dance_video.mp4",      # 댄스 영상
        "sample_video.avi",     # 샘플 영상
        "test.mov",            # 테스트 영상
    ]
    
    print("🎯 YOLO11 비디오 포즈 분석기")
    print("=" * 50)
    
    # 사용자에게 파일 선택하게 하기
    print("분석할 비디오 파일 경로를 입력하세요:")
    print("(예: dance_video.mp4, /path/to/video.mp4)")
    video_path = input("비디오 파일 경로: ").strip()
    
    if not video_path:
        print("❌ 파일 경로가 입력되지 않았습니다.")
    else:
        # 비디오 분석 실행
        results = analyzer.analyze_video(
            video_path=video_path,
            save_output=True,    # 결과 저장
            show_video=True      # 실시간 화면 표시
        )
        
        if results:
            print(f"\n📊 분석 결과:")
            print(f"   - 총 프레임: {results['video_info']['total_frames']}")
            print(f"   - 분석된 프레임: {results['video_info']['analyzed_frames']}")
            print(f"   - 비디오 길이: {results['video_info']['duration']:.2f}초")
            print(f"   - FPS: {results['video_info']['fps']}")
            
            # 포즈가 검출된 프레임 비율 계산
            pose_frames = len(results['pose_data'])
            total_frames = results['video_info']['analyzed_frames']
            if total_frames > 0:
                detection_rate = pose_frames / total_frames * 100
                print(f"   - 포즈 검출률: {detection_rate:.1f}%")
        else:
            print("❌ 비디오 분석에 실패했습니다.")