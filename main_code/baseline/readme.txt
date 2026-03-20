Background section baseline to talk about

python version 3.11.9
--- System Check ---
PyTorch version: 2.5.1+cu121
CUDA Available:  True
GPU Name:       NVIDIA GeForce RTX 4060 Laptop GPU
CUDA Version:    12.1

--- Library Check ---
Ultralytics (YOLO26): 8.4.21
BoxMOT Version:       16.0.10

Dev Log 
>   Using (actiavate the venv)
>   Baseline:
    -> The baseline I'll be evaluating would be YoloV8 and Yolo26 against benchmarks used in papers
    -> Models chosen
        -> Yolo26
            ->Nano
            ->Small
            ->medim
            ->large
            ->x-large
        -> YoloV12
            ->Nano
            ->Small
            ->medim
            ->large
            ->x-large
        -> YoloV8
            ->Nano
            ->Small
            ->medim
            ->large
            ->x-large

    -> Benchmarks chosen
        ->MOT17 for light weight Single object tracking accuracy, Metrics: MOTA, HOTA, IDF1, mAP, IOU
        ->MOT20 for heavy weight Single object tracking accuracy, Metrics: MOTA, HOTA, IDF1, mAP, IOU
        ->ARKitTrack understanding data and IMU noise          ,  Metrics: MOTA, HOTA, IDF1, mAP, IOU
        ->DanceTrack for checking rapid motion                  , Metrics: MOTA, HOTA, IDF1, mAP, IOU
        
    -> Metrics chosen
        ->Higher Order Tracking Accuracy
        ->Multiple Object Tracking Accuracy
        ->Identification F1 score
        ->mAP
        ->Intersection over Union against ground truth. 