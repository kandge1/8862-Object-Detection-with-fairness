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
>   Using YoloV8 Nano, Small and Medium
>   Using Yolo26 Nano, Small and Medium
>   Baseline:
    -> The baseline I'll be evaluating would be YoloV8 and Yolo26 against benchmarks used in papers
    -> Benchmarks chosen
        ->MOT17 for light weight tracking accuracy with metric being HOTA
        ->MOT20 for heavy weight tracking accuracy with metric being HOTA
        ->COCO for for object detecion accuracy with metric being mAP(Also look at mAP over tracking, IOU Againstg ground truth)

>   Run baseline.py
    Baselines are too massive
    I'll first, use it on a sample video of a city block of people walking
    then I'll go to heavier benchmarks for a paper