Background:

In this work, we apply model slicing to YOLO-style detectors, evaluating Subnet‑(r) for 
(r \in {1.0, 0.8, 0.6, 0.4, 0.2}) on COCO and MOT17/MOT20.

Install dependenceis with

.\.venv\Scripts\activate
python -V
python -m pip install --upgrade pip
python -m pip install ultralytics boxmot matplotlib numpy


Dev Log 
>   Using python version 3.12.9(actiavate the venv)
>   Using Ultralytics version '8.4.21'
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