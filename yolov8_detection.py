from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO


# -----------------------------#
# PROJECT PATHS
# -----------------------------#
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = PROJECT_DIR / "videos" / "input" / "objects_video.mp4"
OUTPUT_DIR = PROJECT_DIR / "videos" / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "output_yolov8.mp4"
MODEL_WEIGHTS = "yolov8s.pt"


def main():
    # -----------------------------#
    # CUDA CHECK
    # -----------------------------#
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else "cpu"

    print("CUDA Available:", cuda_available)
    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    # -----------------------------#
    # LOAD PRETRAINED MODEL
    # -----------------------------#
    model = YOLO(MODEL_WEIGHTS)

    # -----------------------------#
    # OPEN RECORDED VIDEO
    # -----------------------------#
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}\n"
            "Place objects_video.mp4 inside videos/input/."
        )

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps_input <= 0:
        fps_input = 30.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), fourcc, fps_input, (width, height)
    )

    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    # -----------------------------#
    # OBJECT DETECTION + FPS
    # -----------------------------#
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.perf_counter()

            results = model(
                frame,
                device=device,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # This measures inference plus annotation time for the frame.
            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            video_writer.write(annotated_frame)

            display_frame = cv2.resize(annotated_frame, (1000, 700))
            cv2.imshow("YOLOv8 Multiple Object Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    print(f"Processed video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
