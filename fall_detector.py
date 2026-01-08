import cv2
from ultralytics import YOLO
import numpy as np
import json
import os
import argparse
import collections

# YOLOv8 포즈 키포인트 매핑 (COCO 데이터셋 기준)
# 0: 코
# 1: 왼쪽 눈
# 2: 오른쪽 눈
# 3: 왼쪽 귀
# 4: 오른쪽 귀
# 5: 왼쪽 어깨
# 6: 오른쪽 어깨
# 7: 왼쪽 팔꿈치
# 8: 오른쪽 팔꿈치
# 9: 왼쪽 손목
# 10: 오른쪽 손목
# 11: 왼쪽 골반
# 12: 오른쪽 골반
# 13: 왼쪽 무릎
# 14: 오른쪽 무릎
# 15: 왼쪽 발목
# 16: 오른쪽 발목

def analyze_video(video_path, output_path, max_frames=None, show_display=False):
    # YOLOv8/11 포즈 모델 로드
    model = YOLO('yolo11n-pose.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 동영상 파일을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 3
    frame_step = max(1, int(fps / target_fps))
    print(f"FPS: {fps}, 1초당 처리할 프레임 수: {target_fps} (Step: {frame_step})")
    
    frame_count = 0
    processed_count = 0
    
    results_data = []
    
    # 쓰러짐 상황 통계
    fall_frames_count = 0
    total_processed_frames = 0
    
    # 속도 계산을 위해 이전 프레임의 골반 Y 좌표 저장
    # 필요하다면 deque를 사용하여 부드럽게 만들 수 있지만, 우선 단순 차이로 계산
    prev_hip_y = None
    
    # 비디오 저장 설정
    out = None
    if show_display:
        output_video_path = output_path.replace('.json', '_annotated.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 저장 FPS는 원본 FPS를 따르거나, 3FPS로 줄여서 저장할 수 있음 (여기선 3FPS로 저장하여 요약본 느낌)
        # 하지만 원본 속도감을 위해 원본 FPS로 저장하되, 처리하지 않은 프레임은 건너뛰므로...
        # -> 처리된 프레임만 이어붙이면 3배속처럼 보일 수 있음. 
        # -> 원본 프레임을 모두 복사하면서 처리된 프레임에만 오버레이? (복잡함)
        # -> 간단히 처리된 프레임만 저장 (요약 영상)
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (w, h))
        print(f"결과 영상 저장 시작: {output_video_path}")
    
    print(f"처리 중: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 지정된 간격마다 프레임 처리 (약 3 FPS)
        if frame_count % frame_step != 0:
            continue
            
        processed_count += 1
        total_processed_frames += 1
        timestamp = frame_count / fps
        
        print(f"프레임 {frame_count} ({timestamp:.2f}s) 처리 중...", end='\r')
            
        if max_frames and processed_count >= max_frames:
            break

        # 감지 수행
        # stream=True는 효율성을 위함이지만, 단순 루프에서는 model(frame)으로 충분
        results = model(frame, verbose=False)
        
        danger_score = 0.0
        pose_state = "알 수 없음 (Unknown)"
        description = "사람이 감지되지 않았습니다."
        velocity = 0.0
        aspect_ratio = 0.0

        # 감지된 객체가 있는지 확인
        if results and len(results) > 0 and results[0].keypoints is not None:
            # 여러 사람이 감지될 경우 첫 번째 사람(인덱스 0)을 추적한다고 가정하거나 가장 위험한 경우를 선택
            # 단순화를 위해 첫 번째 감지된 사람을 사용
            # results[0].keypoints.xy는 (N, 17, 2) 형태의 텐서
            
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            if len(keypoints) > 0:
                person_kps = keypoints[0] # 첫 번째 사람
                
                # 신뢰도 확인 (보통 YOLO conf 임계값에 의해 처리됨)
                
                # 좌표 가져오기
                # 이미지 크기로 정규화? MediaPipe는 정규화하지만, YOLO는 픽셀 좌표를 반환함.
                # 픽셀 좌표를 사용하면 속도는 픽셀/초 단위가 됨.
                # 사람의 키로 정규화하여 스케일 불변으로 만들 수 있음.
                
                h, w, _ = frame.shape
                
                l_hip = person_kps[11]
                r_hip = person_kps[12]
                
                # 골반이 감지되었는지 확인 ([0,0]이 아닌지)
                if np.any(l_hip) and np.any(r_hip):
                    hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0
                    
                    # 높이로 Y 좌표 정규화하여 비교 가능하게 만듦
                    hip_mid_y_norm = hip_mid_y / h
                    
                    if prev_hip_y is not None:
                        # 정규화된 단위/초 속도
                        velocity = (hip_mid_y_norm - prev_hip_y) * fps
                    
                    prev_hip_y = hip_mid_y_norm

                # 종횡비(Aspect Ratio) 계산을 위한 바운딩 박스
                # YOLO results[0].boxes.xyxy는 bbox 좌표 제공
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    if box_h > 0:
                        aspect_ratio = box_w / box_h

                # 휴리스틱 로직 (규칙 기반 판단)
                # 속도 > 0.3 (화면 높이의 약 30%/초) 이면 빠르다고 판단
                velocity_score = 0
                if velocity > 0.3: 
                    velocity_score = min(velocity * 40, 50)
                
                position_score = 0
                if aspect_ratio > 0.8: # 기울어지거나 누운 상태
                    position_score = min(aspect_ratio * 30, 50)

                danger_score = velocity_score + position_score
                
                if aspect_ratio > 1.2:
                    pose_state = "누움 (Lying Down)"
                elif velocity > 0.2:
                    pose_state = "쓰러짐 (Falling)"
                else:
                    pose_state = "서있음/앉음 (Standing/Sitting)"

                # 한국어 행동 해석 (Analysis Result) - 음슴체 ('-함' 체), 배경 설명 제외
                description = ""
                if danger_score >= 30:
                    description = "위험 감지됨. 사람 쓰러짐 또는 급격한 넘어짐 관측됨."
                    fall_frames_count += 1
                elif pose_state == "누움 (Lying Down)":
                    description = "사람 바닥에 누워 있음."
                    fall_frames_count += 1
                elif pose_state == "쓰러짐 (Falling)":
                    description = "사람 빠르게 움직이거나 자세 낮아짐."
                    fall_frames_count += 1
                elif velocity_score > 10:
                     description = "자세 급격히 변함."
                else:
                    description = "특이 사항 없음. 평상시 상태임."

        results_data.append({
            "timestamp": float(round(timestamp, 3)),
            "frame": frame_count,
            "danger_score": float(round(danger_score, 2)),
            "state": pose_state,
            "velocity": float(round(velocity, 4)),
            "aspect_ratio": float(round(aspect_ratio, 4)),
            "description": description
        })
        
        if show_display:
            # 시각화 (Visualization)
            annotated_frame = results[0].plot() if results else frame.copy()
            
            # 정보 텍스트 오버레이
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"State: {pose_state.split('(')[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 한글이 cv2.putText에서 깨질 수 있음
            cv2.putText(annotated_frame, f"Danger: {danger_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 한글 출력을 위해 PIL 사용 등의 복잡함 대신 영문/숫자 정보 위주로 표시하고 
            # 상태값은 간단히만 표시 (cv2는 한글 지원 미비)
            
            # cv2.imshow("CCTV Analysis", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # 비디오 저장 (cv2.imshow 대체)
            if out is not None:
                out.write(annotated_frame)

    cap.release()
    if out is not None:
        out.release()
    # if show_display:
    #     cv2.destroyAllWindows()
    
    # 통계 계산
    # 처리된 프레임 수 기반 (sampled frames)
    # 실제 지속 시간 = (쓰러짐 감지 프레임 수) * (1 / target_fps)  <- 근사치 (샘플링 했으므로)
    # 또는: (쓰러짐 감지 프레임 수) * (원본 FPS의 프레임 간격 * step)
    
    # 프레임 간격(초)
    time_per_step = frame_step / fps
    fall_duration = fall_frames_count * time_per_step
    
    final_output = {
        "summary": {
            "total_frames_analyzed": total_processed_frames,
            "fall_detected": fall_frames_count > 0,
            "fall_duration_seconds": float(round(fall_duration, 2)),
            "fall_frame_count": fall_frames_count,
            "video_fps": fps,
            "analysis_fps": target_fps
        },
        "frames": results_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n분석 완료. 결과가 {output_path} 에 저장되었습니다.")
    print(f"요약: 쓰러짐 감지 {fall_frames_count} 프레임, 약 {fall_duration:.2f}초 지속")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCTV 쓰러짐 감지 분석 (YOLOv8)')
    parser.add_argument('--input', type=str, required=True, help='입력 동영상 파일 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 JSON 파일 경로')
    parser.add_argument('--max_frames', type=int, default=None, help='처리할 최대 프레임 수')
    parser.add_argument('--show', action='store_true', help='분석 화면 시각화')
    
    args = parser.parse_args()
    
    analyze_video(args.input, args.output, args.max_frames, args.show)
