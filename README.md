# YOLOv8 Multiple Object Detection

A Python-based multiple object detection project using the **pretrained Ultralytics YOLOv8s model**. The application reads a recorded MP4 video, processes it frame by frame, and writes an annotated MP4 containing detected objects, bounding boxes, class labels, confidence scores, and an on-screen FPS estimate.

CUDA-enabled PyTorch is used when available; otherwise, inference falls back to the CPU.

## Features

- Multiple-object detection using pretrained `yolov8s.pt` weights.
- Recorded MP4 video as input (not a live camera feed).
- Bounding boxes, class labels, and confidence scores.
- On-screen FPS estimate for each processed frame.
- Annotated MP4 output.
- Automatic CUDA/CPU device selection.

> **Scope:** This repository currently implements video-based object detection only. Object tracking and object counting are listed as future enhancements, not implemented features.

## Repository structure

```text
YOLOv8-Multiple-Object-Detection/
├── assets/
│   └── detection_result_1.png       # Add your detection screenshots
├── yolov8_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

Add your actual video files to the folders before running the code, the input file should be be in "MP4" formate and it will generate the output file in your local directory.

## Requirements

- Python 3.9 or newer (use a Python version supported by the installed PyTorch and Ultralytics releases).
- OpenCV.
- Ultralytics.
- PyTorch.
- Optional: NVIDIA GPU with a compatible NVIDIA driver and CUDA-enabled PyTorch build.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOLOv8-Multiple-Object-Detection.git
cd YOLOv8-Multiple-Object-Detection
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install the appropriate PyTorch build

For NVIDIA GPU acceleration, install the CUDA-enabled PyTorch and torchvision versions that match the current PyTorch installation selector and your system. Follow the official instructions at https://pytorch.org/get-started/locally/.

If you install a CPU-only PyTorch build, the script will use the CPU.

Verify CUDA availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

## Run the project

1. Place your recorded video at `videos/input/objects_video.mp4`.
2. Run:

```bash
python yolov8_detection.py
```

The script loads `yolov8s.pt`. If the weights are not present locally, Ultralytics will normally download them automatically when network access is available. The generated video is saved to:

```text
videos/output/output_yolov8.mp4
```

Press **Q** in the display window to stop processing early.

## How it works

1. Check whether CUDA is available and select GPU or CPU.
2. Load the pretrained YOLOv8s detection model.
3. Open the recorded input video with OpenCV.
4. Read the video frame by frame.
5. Run YOLOv8 inference on each frame.
6. Draw the predicted bounding boxes, class names, and confidence scores.
7. Estimate FPS for the inference-and-annotation step and overlay it on the frame.
8. Write the annotated frames to the output MP4.

## Example results

![YOLOv8 detection result](https://github.com/user-attachments/assets/47a77a5a-7f43-4d30-ad9b-086629449fb1)


The screenshots should be taken from your actual model output. Do not report illustrative confidence values or FPS values as benchmark results.

## Hardware used

The development system used for this project includes:

- GPU: NVIDIA GeForce MX330
- Framework: PyTorch with CUDA support when available
- Model: pretrained YOLOv8s

The displayed FPS depends on video resolution, GPU/CPU configuration, drivers, software versions, and the workload. It is an observed per-frame processing estimate, not a standardized benchmark.

## Future enhancements

- Real-time detection using laptop and mobile cameras.
- Multi-object tracking with persistent object IDs.
- Object counting based on tracking or virtual counting lines.
- Further inference and deployment optimization.

## Author

**Vaibhav Govindwar**  
M.Tech, Data Science  
GITAM School of Technology, Hyderabad
