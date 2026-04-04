# In this version:
# Use local images as uploaded frames
# Run real YOLO26n inference
# Ouput results as annotated images 

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import time

import cv2
from ultralytics import YOLO


# 1. Define client state
@dataclass
class ClientState:
    user_id: int
    current_frame_id: int
    tracking_age: int
    last_offload_frame_id: int


# 2. Define scheduler output
@dataclass
class ScheduleDecision:
    user_id: int
    offload_period: int


# 3. Define inference request
@dataclass
class InferenceRequest:
    request_id: int
    user_id: int
    frame_id: int
    frame_path: str
    arrival_time: int


# 4. Define scheduler
class Scheduler:
    def decide(self, client_state: ClientState) -> ScheduleDecision:
        # Placeholder decision logic
        return ScheduleDecision(
            user_id=client_state.user_id,
            offload_period=3
        )


# 5. Define client simulator
class ClientSimulator:
    def __init__(self, user_id: int, frame_paths: List[str]) -> None:
        self.user_id = user_id
        self.frame_paths = frame_paths
        self.total_frames = len(frame_paths)
        self.current_frame_id = 0

        self.tracking_age = 0
        self.last_offload_frame_id = -1
        self.offload_period = 3

    def get_state(self) -> ClientState:
        return ClientState(
            user_id=self.user_id,
            current_frame_id=self.current_frame_id,
            tracking_age=self.tracking_age,
            last_offload_frame_id=self.last_offload_frame_id
        )

    def apply_schedule(self, decision: ScheduleDecision) -> None:
        self.offload_period = decision.offload_period

    def maybe_generate_request(
        self,
        request_id: int,
        current_time: int
    ) -> Optional[InferenceRequest]:
        if self.current_frame_id >= self.total_frames:
            return None

        should_offload = (self.current_frame_id % self.offload_period == 0)

        if should_offload:
            req = InferenceRequest(
                request_id=request_id,
                user_id=self.user_id,
                frame_id=self.current_frame_id,
                frame_path=self.frame_paths[self.current_frame_id],
                arrival_time=current_time
            )
            self.last_offload_frame_id = self.current_frame_id
            self.tracking_age = 0
            return req

        self.tracking_age += 1
        return None

    def step(self) -> None:
        self.current_frame_id += 1


# 6. Define queue
class RequestQueue:
    def __init__(self) -> None:
        self.items: List[InferenceRequest] = []

    def push(self, request: InferenceRequest) -> None:
        self.items.append(request)

    def pop(self) -> Optional[InferenceRequest]:
        if not self.items:
            return None
        return self.items.pop(0)

    def size(self) -> int:
        return len(self.items)

    def is_empty(self) -> bool:
        return len(self.items) == 0


# 7. Build batch
# Just a simple batch frame: trigger inference when the number of queued requests reaches the predefined batch size
def build_batch(queue: RequestQueue, batch_size: int) -> List[InferenceRequest]:
    batch: List[InferenceRequest] = []

    for _ in range(batch_size):
        req = queue.pop()
        if req is None:
            break
        batch.append(req)

    return batch


# 8. Model selector
class ModelSelector:
    def select_model(self, info: Dict[str, int]) -> str:
        # For now, always use YOLO26n as requested
        return "yolo26n"


# 9. YOLO detector
class YOLODetector:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def infer_batch(self, image_paths: List[str]):
        # verbose=False to suppress terminal output
        return self.model(image_paths, verbose=False)


# 10. Result helpers
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_detection_summary(result) -> Dict[str, Any]:
    boxes = result.boxes

    detections = []
    if boxes is not None:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names.get(cls_id, str(cls_id))

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox_xyxy": [float(v) for v in xyxy],
            })

    return {
        "image_path": result.path,
        "num_detections": len(detections),
        "detections": detections,
    }


def save_annotated_image(result, save_path: Path) -> None:
    plotted = result.plot()
    cv2.imwrite(str(save_path), plotted)


# 11. Batch inference execution
def run_batch_inference(
    batch: List[InferenceRequest],
    detector: YOLODetector,
    output_dir: Path
) -> List[Dict[str, Any]]:
    image_paths = [req.frame_path for req in batch]
    results = detector.infer_batch(image_paths)

    batch_records: List[Dict[str, Any]] = []

    for req, result in zip(batch, results):
        request_dir = output_dir / f"user_{req.user_id}" / f"frame_{req.frame_id:04d}"
        ensure_dir(request_dir)

        summary = extract_detection_summary(result)
        summary["request"] = asdict(req)

        annotated_path = request_dir / "annotated.jpg"
        save_annotated_image(result, annotated_path)

        json_path = request_dir / "result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        batch_records.append({
            "request_id": req.request_id,
            "user_id": req.user_id,
            "frame_id": req.frame_id,
            "frame_path": req.frame_path,
            "annotated_image": str(annotated_path),
            "result_json": str(json_path),
            "num_detections": summary["num_detections"],
        })

    return batch_records


# 12. Utility: load frames from folder
def load_frame_paths(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in exts
    )
    return paths


# 13. Main
def main() -> None:
    model_path = "yolo26n.pt"
    output_dir = Path("outputs")
    ensure_dir(output_dir)

    # Prepare local frame sequences
    user1_frames = load_frame_paths("frames/user1")
    user2_frames = load_frame_paths("frames/user2")

    if len(user1_frames) == 0 or len(user2_frames) == 0:
        raise FileNotFoundError(
            "No input images found. Please put images into frames/user1 and frames/user2."
        )

    scheduler = Scheduler()
    queue = RequestQueue()
    model_selector = ModelSelector()
    detector = YOLODetector(model_path=model_path)

    clients = [
        ClientSimulator(user_id=1, frame_paths=user1_frames),
        ClientSimulator(user_id=2, frame_paths=user2_frames),
    ]

    next_request_id = 0
    batch_size = 2
    total_steps = max(len(user1_frames), len(user2_frames))

    run_summary: Dict[str, Any] = {
        "model": "yolo26n",
        "batch_size": batch_size,
        "total_steps": total_steps,
        "batches": [],
    }

    for current_time in range(total_steps):
        for client in clients:
            if client.current_frame_id >= client.total_frames:
                continue

            state = client.get_state()
            decision = scheduler.decide(state)
            client.apply_schedule(decision)

        for client in clients:
            req = client.maybe_generate_request(
                request_id=next_request_id,
                current_time=current_time
            )
            if req is not None:
                queue.push(req)
                next_request_id += 1

        if queue.size() >= batch_size:
            batch = build_batch(queue, batch_size=batch_size)

            selector_input = {
                "queue_length": len(batch),
                "batch_size": len(batch),
                "active_users": len(clients),
            }
            selected_model = model_selector.select_model(selector_input)

            if selected_model != "yolo26n":
                raise ValueError("This script is configured to run YOLO26n only.")

            t0 = time.time()
            batch_records = run_batch_inference(
                batch=batch,
                detector=detector,
                output_dir=output_dir
            )
            t1 = time.time()

            run_summary["batches"].append({
                "time_step": current_time,
                "batch_size": len(batch),
                "latency_sec": t1 - t0,
                "records": batch_records,
            })

        for client in clients:
            if client.current_frame_id < client.total_frames:
                client.step()

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()