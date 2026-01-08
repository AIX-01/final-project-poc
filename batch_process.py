import os
import glob
import subprocess
import argparse

def batch_process(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    print(f"{input_dir} 에서 {len(video_files)} 개의 동영상을 발견했습니다.")
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        json_name = os.path.splitext(video_name)[0] + "_analysis.json"
        output_path = os.path.join(output_dir, json_name)
        
        print(f"{video_name} 처리 중...")
        
        # fall_detector.py 호출
        # 이 스크립트와 동일한 디렉토리에 fall_detector.py가 있다고 가정
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fall_detector.py")
        
        cmd = ["python", script_path, "--input", video_path, "--output", output_path]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"{video_name} 처리 중 오류 발생: {e}")
            
    print("일괄 처리 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCTV 동영상 일괄 처리')
    parser.add_argument('--input_dir', type=str, required=True, help='동영상 파일이 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True, help='JSON 결과를 저장할 디렉토리')
    
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output_dir)
