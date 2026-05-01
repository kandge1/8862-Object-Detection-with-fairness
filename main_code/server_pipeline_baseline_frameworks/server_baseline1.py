# Baseline1: model pool loading, est from client, same-opt_model batching
import json
import time
import asyncio
import heapq
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch

# ============================================================
# 1. Constants and data structures
# ============================================================

# Python 3.8 compatible typing: use Dict/List instead of dict[]/list[].
mAP_table: Dict[str, Dict[str, float]] = {
    "yolov8n_288": {"map": 0.3222, "time": 1.9438},
    "yolov8s_288": {"map": 0.3879, "time": 2.0587},
    "yolov8m_288": {"map": 0.4425, "time": 2.3455},
    "yolov8l_288": {"map": 0.4578, "time": 2.7165},
    "yolov8x_288": {"map": 0.4699, "time": 3.6420},
    "yolov8n_416": {"map": 0.4412, "time": 2.5249},
    "yolov8s_416": {"map": 0.4899, "time": 2.3857},
    "yolov8m_416": {"map": 0.5200, "time": 2.9933},
    "yolov8l_416": {"map": 0.5211, "time": 4.0068},
    "yolov8x_416": {"map": 0.5357, "time": 5.3583},
    "yolov8n_640": {"map": 0.5433, "time": 3.0423},
    "yolov8s_640": {"map": 0.5838, "time": 3.3686},
    "yolov8m_640": {"map": 0.5933, "time": 5.0350},
    "yolov8l_640": {"map": 0.5846, "time": 7.0360},
    "yolov8x_640": {"map": 0.6017, "time": 10.1924},
    "yolov8n_864": {"map": 0.5871, "time": 3.7886},
    "yolov8s_864": {"map": 0.6183, "time": 4.4741},
    "yolov8m_864": {"map": 0.6143, "time": 7.7156},
    "yolov8l_864": {"map": 0.6148, "time": 11.0286},
    "yolov8x_864": {"map": 0.6276, "time": 16.8240},
    "yolov8n_1088": {"map": 0.6041, "time": 4.7502},
    "yolov8s_1088": {"map": 0.6316, "time": 6.4990},
    "yolov8m_1088": {"map": 0.6300, "time": 11.4111},
    "yolov8l_1088": {"map": 0.6305, "time": 16.2437},
    "yolov8x_1088": {"map": 0.6360, "time": 24.8706},
}

# Model pool: preload yolo26 n/s/m/l/x onto GPU at program start.
# Override with env, e.g.:
#   MODEL_POOL=yolo26n:yolo26n.pt,yolo26s:yolo26s.pt python3 server_pipeline_v4_model_pool.py
DEFAULT_MODEL_POOL_SPEC = "yolo26n:yolo26n.pt,yolo26s:yolo26s.pt,yolo26m:yolo26m.pt,yolo26l:yolo26l.pt,yolo26x:yolo26x.pt"
MODEL_POOL_SPEC = os.environ.get("MODEL_POOL", DEFAULT_MODEL_POOL_SPEC)
DEFAULT_SERVER_MODEL_NAME = os.environ.get("DEFAULT_SERVER_MODEL_NAME", "yolo26n")
DEFAULT_SERVER_IMGSZ = int(os.environ.get("DEFAULT_SERVER_IMGSZ", "640"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
DEVICE = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class InferenceRequest:
    request_id: int
    user_id: int
    frame_id: int
    frame_path: str
    arrival_time: float
    opt_model: str
    est_map: float
    est_time: float
    tracking_age: int
    capture_time: float = 0.0
    client_fps: float = 0.0

    @property
    def response_key(self) -> str:
        # request_id may restart from 0 for each client, so include user_id.
        return f"u{self.user_id}_r{self.request_id}"


# ============================================================
# 2. Queue and batch scheduling logic
# ============================================================

class PriorityRequestQueue:
    def __init__(self, age_penalty: float = 0.01) -> None:
        self.heap: List[Tuple[float, float, int, int, InferenceRequest]] = []
        self.age_penalty = age_penalty

    def push(self, request: InferenceRequest, current_time: float) -> None:
        # Current test mode: FIFO by arrival time. Keeps behavior simple.
        dynamic_priority = request.arrival_time
        heapq.heappush(
            self.heap,
            (dynamic_priority, request.arrival_time, request.user_id, request.request_id, request),
        )

    def pop_matching(self, target_model: str) -> Optional[InferenceRequest]:
        matching_items = [item for item in self.heap if item[4].opt_model == target_model]
        if not matching_items:
            return None
        best_match = min(matching_items, key=lambda x: x[0])
        self.heap.remove(best_match)
        heapq.heapify(self.heap)
        return best_match[4]

    def pop(self) -> Optional[InferenceRequest]:
        if not self.heap:
            return None
        return heapq.heappop(self.heap)[4]

    def get_sum_map(self, model: str, b1_time: float) -> Tuple[float, float]:
        if not self.heap:
            return 0.0, 0.0
        if model not in mAP_table:
            model = "yolov8n_640"

        total_map = 0.0
        map_square = 0.0
        model_time = mAP_table[model]["time"]
        total_est_time = len(self.heap) * model_time + b1_time

        for item in self.heap:
            base_map = item[4].est_map
            penalized_map = max(0.0, base_map - (total_est_time * self.age_penalty))
            total_map += penalized_map
            map_square += penalized_map ** 2
        return total_map, map_square

    def get_opt_model(self) -> str:
        if not self.heap:
            return "yolov8n_640"
        sum_map = sum(item[4].est_map for item in self.heap)
        avg_map = sum_map / len(self.heap)
        return min(mAP_table.keys(), key=lambda model_name: abs(mAP_table[model_name]["map"] - avg_map))

    def size(self) -> int:
        return len(self.heap)

    def is_empty(self) -> bool:
        return len(self.heap) == 0


def _safe_model_key(model: str) -> str:
    return model if model in mAP_table else "yolov8n_640"


def calculate_sum_map(batch: List[InferenceRequest], model: str, age_penalty: float = 0.01) -> Tuple[float, float, float]:
    if not batch:
        return 0.0, 0.0, 0.0
    model = _safe_model_key(model)
    total_map = 0.0
    map_square = 0.0
    model_time = mAP_table[model]["time"]
    total_est_time = len(batch) * model_time

    for req in batch:
        penalized_map = max(0.0, req.est_map - (total_est_time * age_penalty))
        total_map += penalized_map
        map_square += penalized_map ** 2
    return total_map, map_square, total_est_time


def batch_get_opt_model(batch: List[InferenceRequest]) -> str:
    if not batch:
        return "yolov8n_640"
    avg_map = sum(item.est_map for item in batch) / len(batch)
    return min(mAP_table.keys(), key=lambda model_name: abs(mAP_table[model_name]["map"] - avg_map))


def build_batch(queue: PriorityRequestQueue) -> List[InferenceRequest]:
    """
    Baseline1 batch construction.

    Keep the original queue pop policy: the first request is still the earliest-arrived
    request according to PriorityRequestQueue.pop(). Then batch only requests with the
    exact same opt_model string, e.g. yolov8n_640 only batches with yolov8n_640.

    No avg-est-map / fairness scoring is used here, and no server-side model
    re-selection happens during batch construction.
    """
    batch: List[InferenceRequest] = []
    if queue.is_empty():
        return batch

    first_req = queue.pop()
    if first_req is None:
        return batch

    target_opt_model = first_req.opt_model
    batch.append(first_req)

    remaining: List[InferenceRequest] = []

    while not queue.is_empty():
        candidate = queue.pop()
        if candidate is None:
            break

        if candidate.opt_model == target_opt_model:
            batch.append(candidate)
        else:
            remaining.append(candidate)

    for req in remaining:
        queue.push(req, req.arrival_time)

    return batch


# ============================================================
# 3. YOLO inference utilities
# ============================================================

def parse_model_pool_spec(spec: str) -> Dict[str, str]:
    pool: Dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            path = item
            name = Path(path).stem
        pool[name.strip()] = path.strip()
    return pool


def model_name_from_opt_model(opt_model: str) -> str:
    # Example: yolov8n_640 -> yolo26n. Falls back to default if parsing fails.
    match = re.match(r"yolov8([nslmx])_\d+", opt_model or "")
    if not match:
        return DEFAULT_SERVER_MODEL_NAME
    return f"yolo26{match.group(1)}"


def imgsz_from_opt_model(opt_model: str) -> int:
    # Example: yolov8n_640 -> 640. Falls back to default if parsing fails.
    match = re.match(r"yolov8[nslmx]_(\d+)", opt_model or "")
    if not match:
        return DEFAULT_SERVER_IMGSZ
    return int(match.group(1))


class YOLOModelPool:
    def __init__(self, pool_spec: str, device: str) -> None:
        self.device = device
        self.model_paths = parse_model_pool_spec(pool_spec)
        if not self.model_paths:
            raise ValueError("MODEL_POOL is empty; provide at least one model_name:model_path pair.")

        self.models: Dict[str, YOLO] = {}
        print(f"Loading YOLO model pool on device={device}...")
        for name, model_path in self.model_paths.items():
            print(f"  loading {name}: {model_path}")
            model = YOLO(model_path)
            model.to(device)
            self.models[name] = model
        print(f"Model pool ready: {list(self.models.keys())}")

    def resolve_model_name(self, opt_model: str) -> str:
        requested = model_name_from_opt_model(opt_model)
        if requested in self.models:
            return requested
        if DEFAULT_SERVER_MODEL_NAME in self.models:
            return DEFAULT_SERVER_MODEL_NAME
        return next(iter(self.models.keys()))

    def infer_batch(self, model_name: str, image_paths: List[str], imgsz: int = 640):
        model_name = model_name if model_name in self.models else self.resolve_model_name(model_name)
        return self.models[model_name](image_paths, imgsz=imgsz, conf=CONF_THRESHOLD, device=self.device, verbose=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_image_shape(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        return {"height": None, "width": None, "channels": None, "shape_hw": None}
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    return {"height": int(h), "width": int(w), "channels": int(c), "shape_hw": [int(h), int(w)]}


def extract_detection_summary(result, source_image_path: str) -> Dict[str, Any]:
    """
    Return KCF-ready detection data.
    KCF needs bbox in xywh: (x, y, width, height).
    Client can still use bbox_xyxy for drawing red server boxes.
    """
    image_info = _read_image_shape(source_image_path)
    img_w = image_info.get("width")
    img_h = image_info.get("height")

    boxes = result.boxes
    detections: List[Dict[str, Any]] = []

    if boxes is not None:
        for i in range(len(boxes)):
            xyxy_raw = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy_raw]

            # Clamp to image bounds if available. KCF dislikes invalid boxes.
            if img_w is not None and img_h is not None:
                x1 = max(0.0, min(x1, float(img_w - 1)))
                y1 = max(0.0, min(y1, float(img_h - 1)))
                x2 = max(0.0, min(x2, float(img_w - 1)))
                y2 = max(0.0, min(y2, float(img_h - 1)))

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names.get(cls_id, str(cls_id))
            is_valid_for_kcf = bool(w >= 2.0 and h >= 2.0 and conf >= CONF_THRESHOLD)

            detections.append({
                "track_id": None,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xyxy_int": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "bbox_xywh": [x1, y1, w, h],
                "bbox_xywh_int": [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))],
                "area": float(w * h),
                "is_valid_for_kcf": is_valid_for_kcf,
                "source": "server_yolo_detection",
            })

    return {
        "image_path": result.path,
        "image": image_info,
        "num_detections": len(detections),
        "detections": detections,
    }


def save_annotated_image(result, save_path: Path) -> None:
    plotted = result.plot()
    cv2.imwrite(str(save_path), plotted)


# ============================================================
# 4. FastAPI server application
# ============================================================

app = FastAPI(title="Edge-to-Cloud YOLO Server - KCF Ready")

global_queue = PriorityRequestQueue(age_penalty=0.01)
model_pool = YOLOModelPool(MODEL_POOL_SPEC, DEVICE)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "server_uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "server_outputs"))
ensure_dir(UPLOAD_DIR)
ensure_dir(OUTPUT_DIR)

pending_responses: Dict[str, asyncio.Future] = {}
_batch_counter = 0


async def _resolve_future(req: InferenceRequest, payload: Dict[str, Any]) -> None:
    future = pending_responses.pop(req.response_key, None)
    if future and not future.done():
        future.set_result(payload)


async def batch_processing_worker():
    global _batch_counter
    print("Background batch processing worker started...")
    print(f"Inference backend: model_pool={list(model_pool.models.keys())}, device={DEVICE}")

    while True:
        if not global_queue.is_empty():
            batch = build_batch(global_queue)
            if not batch:
                await asyncio.sleep(0.01)
                continue

            _batch_counter += 1
            batch_id = _batch_counter
            inference_start = time.time()
            image_paths = [req.frame_path for req in batch]

            # Baseline1: after batching same opt_model requests, directly use that
            # first request's opt_model to choose model size and resolution.
            batch_opt_model = batch[0].opt_model
            actual_model_name = model_pool.resolve_model_name(batch_opt_model)
            actual_imgsz = imgsz_from_opt_model(batch_opt_model)

            try:
                results = model_pool.infer_batch(actual_model_name, image_paths, imgsz=actual_imgsz)
                inference_finish = time.time()
            except Exception as exc:
                inference_finish = time.time()
                for req in batch:
                    await _resolve_future(req, {
                        "status": "error",
                        "error": "inference_failed",
                        "detail": str(exc),
                        "request_id": req.request_id,
                        "user_id": req.user_id,
                        "frame_id": req.frame_id,
                    })
                await asyncio.sleep(0.01)
                continue

            batch_request_ids = [req.request_id for req in batch]
            batch_user_ids = [req.user_id for req in batch]

            for req, result in zip(batch, results):
                request_dir = OUTPUT_DIR / f"user_{req.user_id}" / f"frame_{req.frame_id:06d}"
                ensure_dir(request_dir)

                summary = extract_detection_summary(result, req.frame_path)
                summary["request"] = asdict(req)
                summary["server_model"] = actual_model_name
                summary["server_model_path"] = model_pool.model_paths.get(actual_model_name)
                summary["server_imgsz"] = actual_imgsz

                annotated_path = request_dir / "annotated.jpg"
                save_annotated_image(result, annotated_path)

                json_path = request_dir / "result.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

                now = time.time()
                client_response = {
                    "status": "success",
                    "request_id": req.request_id,
                    "user_id": req.user_id,
                    "frame_id": req.frame_id,

                    # What the client originally requested/hinted.
                    "client_hint": {
                        "opt_model": req.opt_model,
                        "est_map": req.est_map,
                        "est_time": req.est_time,
                        "tracking_age": req.tracking_age,
                        "client_fps": req.client_fps,
                        "capture_time": req.capture_time,
                    },

                    # What server actually used now.
                    "server_inference": {
                        "model_name": actual_model_name,
                        "model_path": model_pool.model_paths.get(actual_model_name),
                        "imgsz": actual_imgsz,
                        "conf_threshold": CONF_THRESHOLD,
                    },

                    "batch": {
                        "batch_id": batch_id,
                        "batch_size": len(batch),
                        "request_ids": batch_request_ids,
                        "user_ids": batch_user_ids,
                    },

                    "timing": {
                        "arrival_time": req.arrival_time,
                        "inference_start_time": inference_start,
                        "inference_finish_time": inference_finish,
                        "response_time": now,
                        "queue_wait_sec": inference_start - req.arrival_time,
                        "batch_inference_latency_sec": inference_finish - inference_start,
                        "server_total_latency_sec": now - req.arrival_time,
                    },

                    # Useful to debug paths. annotated_image_path is on SERVER machine.
                    "paths": {
                        "server_input_image_path": req.frame_path,
                        "annotated_image_path": str(annotated_path),
                        "result_json": str(json_path),
                    },

                    # KCF-ready fields.
                    "image": summary["image"],
                    "num_detections": summary["num_detections"],
                    "detections": summary["detections"],
                }

                await _resolve_future(req, client_response)

        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_processing_worker())


@app.post("/infer")
async def receive_inference_request(
    request_id: int = Form(...),
    user_id: int = Form(...),
    frame_id: int = Form(...),
    opt_model: str = Form(...),
    est_map: float = Form(...),
    est_time: float = Form(...),
    tracking_age: int = Form(...),
    capture_time: float = Form(0.0),
    client_fps: float = Form(0.0),
    file: UploadFile = File(...),
):
    arrival_time = time.time()

    safe_name = Path(file.filename or f"frame_{frame_id}.jpg").name
    file_location = UPLOAD_DIR / f"req{request_id}_u{user_id}_f{frame_id}_{safe_name}"
    with open(file_location, "wb+") as f:
        f.write(await file.read())

    req = InferenceRequest(
        request_id=request_id,
        user_id=user_id,
        frame_id=frame_id,
        frame_path=str(file_location),
        arrival_time=arrival_time,
        opt_model=opt_model,
        est_map=est_map,
        est_time=est_time,
        tracking_age=tracking_age,
        capture_time=capture_time,
        client_fps=client_fps,
    )

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending_responses[req.response_key] = future
    global_queue.push(req, arrival_time)

    try:
        result = await future
        return JSONResponse(content=result)
    except asyncio.CancelledError:
        pending_responses.pop(req.response_key, None)
        return JSONResponse(status_code=499, content={"error": "Client Closed Request"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "queue_size": global_queue.size(),
        "pending_responses": len(pending_responses),
        "model_pool": model_pool.model_paths,
        "default_server_model_name": DEFAULT_SERVER_MODEL_NAME,
        "default_server_imgsz": DEFAULT_SERVER_IMGSZ,
        "device": DEVICE,
        "conf_threshold": CONF_THRESHOLD,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
