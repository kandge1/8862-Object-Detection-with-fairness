import os
import json
import time
import asyncio
import heapq
import cv2
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# 1. CONSTANTS & DATA STRUCTURES

# Pre-computed lookup table for bechmark of different model sizes and resolutions.
mAP_table: dict[str, dict[str, float]] = {
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

@dataclass
class InferenceRequest:
    """Standardized object to hold all data related to a single uploaded frame."""
    request_id: int
    user_id: int
    frame_id: int
    frame_path: str
    arrival_time: float
    opt_model: str
    est_map: float
    est_time: float

# 2. QUEUE & SCHEDULING LOGIC

class PriorityRequestQueue:
    """
    A custom Priority Queue that orders items not just by their baseline mAP,
    but dynamically applies an "age penalty" so older requests naturally rise
    to the front of the line.
    """
    def __init__(self, age_penalty: float = 0.01) -> None:
        # Format: (priority_score, arrival_time, request_id, request_object)
        # low priority_score request moves front
        self.heap: List[Tuple[float, float, int, InferenceRequest]] = []
        self.age_penalty = age_penalty

    def push(self, request: InferenceRequest, current_time: float) -> None:
        """Calculates dynamic priority and pushes onto the min-heap."""
        # # consider wait time
        # wait_time = current_time - request.arrival_time
        # dynamic_priority = request.est_map - (wait_time * self.age_penalty)

        # First in first out
        dynamic_priority = request.arrival_time
        heapq.heappush(
            self.heap, 
            (dynamic_priority, request.arrival_time, request.request_id, request)
        )

    def pop_matching(self, target_model: str) -> Optional['InferenceRequest']:
        """Searches the queue for the same model request as first request."""
        matching_items = [item for item in self.heap if item[3].opt_model == target_model]
        if not matching_items:
            return None
        
        # Min function pulls the tuple with the lowest priority_score (item[0])
        best_match = min(matching_items, key=lambda x: x[0])       
        self.heap.remove(best_match)
        # Re-sort the heap after manual removal to maintain the min-heap property
        heapq.heapify(self.heap)
        
        # Return just the InferenceRequest object (index 3)
        return best_match[3]

    def pop(self) -> Optional[InferenceRequest]:
        """Pulls the absolutely most urgent request from the front of the heap."""
        if not self.heap:
            return None
        return heapq.heappop(self.heap)[3]
    
    def get_sum_map(self, model: str, b1_time: float) -> Tuple[float, float]:
        """Calculates what the queue's mAP totals of batch 2."""
        if not self.heap:
            return 0.0, 0.0
        
        total_map = 0.0
        map_square = 0.0
        model_time = mAP_table[model]["time"]
        
        # batch 1+2 time
        total_est_time = len(self.heap) * model_time + b1_time

        for item in self.heap: 
            base_map = item[3].est_map
            # Apply the penalty based on the projected future time
            penalized_map = max(0.0, base_map - (total_est_time * self.age_penalty))
            
            total_map += penalized_map
            map_square += penalized_map ** 2
            
        return total_map, map_square

    def get_opt_model(self) -> str:
        """Finds the model whose baseline mAP most closely aligns with the queue's average."""
        if not self.heap:
            return "yolov8n_288" 
            
        sum_map = sum(item[3].est_map for item in self.heap)
        avg_map = sum_map / len(self.heap)

        best_model = min(
            mAP_table.keys(), 
            key=lambda model_name: abs(mAP_table[model_name]["map"] - avg_map)
        )
        return best_model

    def size(self) -> int:
        return len(self.heap)

    def is_empty(self) -> bool:
        return len(self.heap) == 0


def calculate_sum_map(
    batch: List['InferenceRequest'], 
    model: str,  
    age_penalty: float = 0.01
) -> Tuple[float, float, float]:
    """Helper function to calculate sums specifically for a built batch list."""
    if not batch:
        return 0.0, 0.0, 0.0
    
    total_map = 0.0
    map_square = 0.0
    model_time = mAP_table[model]["time"]
    total_est_time = len(batch) * model_time

    for req in batch:
        base_map = req.est_map
        penalized_map = max(0.0, base_map - (total_est_time * age_penalty))
        
        total_map += penalized_map
        map_square += penalized_map ** 2   
    
    return total_map, map_square, total_est_time


def batch_get_opt_model(batch: List['InferenceRequest']) -> str:
    """Helper function to find the optimal model for a built batch list."""
    if not batch:
        return "yolov8n_288" 
        
    sum_map = sum(item.est_map for item in batch)
    avg_map = sum_map / len(batch)

    best_model = min(
        mAP_table.keys(), 
        key=lambda model_name: abs(mAP_table[model_name]["map"] - avg_map)
    )
    return best_model


def build_batch(queue: PriorityRequestQueue) -> List['InferenceRequest']:
    """
    Core Scheduling Algorithm: Builds a batch by pulling from the queue, calculating
    the resulting average mAP and Jain's Index, and stopping when adding more items would 
    reduce the overall system score.
    """
    batch: List['InferenceRequest'] = []

    if queue.is_empty():
        return batch  
    
    total_size = queue.size() 
    
    # 1. Seed the batch with the most urgent request
    first_req = queue.pop()
    b1_model = first_req.opt_model
    b2_model = queue.get_opt_model()
    batch.append(first_req)

    # Calculate initial score based on 1 item in batch
    b1_map, b1_map_square, b1_time = calculate_sum_map(batch, b1_model) 
    b2_map, b2_map_square = queue.get_sum_map(b2_model, b1_time)
    avg_map = (b1_map + b2_map) / total_size
    
    # Jain's Fairness Index formula: (Sum X)^2 / (N * Sum X^2)
    denominator = total_size * (b1_map_square + b2_map_square)
    jain = ((b1_map + b2_map) ** 2 / denominator) if denominator > 0 else 1.0

    current_score = avg_map + jain

    # 2. Iteratively test adding more items to the batch (greedy algorithm)
    while not queue.is_empty():
        # Prefer items that want the same model as our seed to avoid model switching costs
        candidate = queue.pop_matching(first_req.opt_model)
        
        if candidate is None:
            candidate = queue.pop()

        # Create a temporary batch to test
        test_batch = batch + [candidate]
        b1_model = batch_get_opt_model(test_batch)
        b2_model = queue.get_opt_model()

        b1_map, b1_map_square, b1_time = calculate_sum_map(test_batch, b1_model)
        b2_map, b2_map_square = queue.get_sum_map(b2_model, b1_time)
        avg_map = (b1_map + b2_map) / total_size
        
        test_denom = total_size * (b1_map_square + b2_map_square)
        jain = ((b1_map + b2_map) ** 2 / test_denom) if test_denom > 0 else 1.0
        test_score = avg_map + jain

        # 3. Decision point: Keep it or put it back
        if test_score > current_score:
            batch.append(candidate)
            current_score = test_score
        else:
            # Adding this item reduced the score, so we reject it.
            # Must pass its original arrival time back so it keeps its old age priority!
            # Use arrival time for easy computation
            queue.push(candidate, candidate.arrival_time) 
            break

    return batch


# 3. YOLO INFERENCE UTILITIES
class YOLODetector:
    """Wrapper class to handle loading and executing the PyTorch YOLO model."""
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def infer_batch(self, image_paths: List[str]):
        # verbose=False suppresses standard YOLO terminal spam
        return self.model(image_paths, verbose=False)


def ensure_dir(path: Path) -> None:
    """Helper to safely create nested directories if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)

def extract_detection_summary(result) -> Dict[str, Any]:
    """Parses raw YOLO results into a clean JSON dictionary."""
    boxes = result.boxes
    detections = []
    
    if boxes is not None:
        for i in range(len(boxes)):
            # Convert tensors to standard Python types for JSON serialization
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
    """Draws bounding boxes onto the image and saves it to disk."""
    plotted = result.plot()
    cv2.imwrite(str(save_path), plotted)


# 4. FASTAPI SERVER APPLICATION
app = FastAPI(title="Edge-to-Cloud YOLO Server")

# Global Queue instance shared across all incoming connections
global_queue = PriorityRequestQueue(age_penalty=0.01)
# Initialize the model once in memory at startup
detector = YOLODetector("yolov8n.pt") 

# Directories for physical file storage
UPLOAD_DIR = Path("server_uploads")
OUTPUT_DIR = Path("server_outputs")
ensure_dir(UPLOAD_DIR)
ensure_dir(OUTPUT_DIR)

# Important Async Dictionary: 
# Maps a specific request_id to an asyncio.Future object. 
# This is the "bridge" that allows the HTTP endpoint to pause, while the 
# background worker processes the GPU task, and then pass the result back.
pending_responses: Dict[int, asyncio.Future] = {}


async def batch_processing_worker():
    """
    This function runs in a continuous infinite loop in the background.
    It acts as the single 'consumer' of the queue, ensuring GPU resources
    are used optimally via batching, without blocking incoming HTTP requests.
    """
    print("Background batch processing worker started...")
    
    while True:
        if not global_queue.is_empty():
            # 1. Ask the scheduling algorithm to group waiting frames
            batch = build_batch(global_queue)
            
            # 2. Run the heavy PyTorch inference
            image_paths = [req.frame_path for req in batch]
            results = detector.infer_batch(image_paths)

            # 3. Process the results for each frame individually
            for req, result in zip(batch, results):
                request_dir = OUTPUT_DIR / f"user_{req.user_id}" / f"frame_{req.frame_id:04d}"
                ensure_dir(request_dir)

                # Save JSON metadata and annotated image locally to the server
                summary = extract_detection_summary(result)
                summary["request"] = asdict(req)
                
                annotated_path = request_dir / "annotated.jpg"
                save_annotated_image(result, annotated_path)
                
                json_path = request_dir / "result.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

                # Format the specific response payload the client needs
                client_response = {
                    "status": "success",
                    "request_id": req.request_id,
                    "annotated_image_path": str(annotated_path),
                    "num_detections": summary["num_detections"],
                    "detections": summary["detections"]
                }

                # 4. Resolve the HTTP Future!
                # We look up the specific Future object tied to this request_id.
                # Calling set_result() instantly unpauses the HTTP endpoint waiting for it.
                future = pending_responses.pop(req.request_id, None)
                if future and not future.done():
                    future.set_result(client_response)

        # Critical: If the queue is empty (or after a batch finishes), yield control
        # back to the FastAPI event loop for 10ms. If we don't do this, the while True 
        # loop will lock up the entire CPU core and block new HTTP requests.
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup_event():
    """Triggered automatically by FastAPI when uvicorn starts the server."""
    asyncio.create_task(batch_processing_worker())


@app.post("/infer")
async def receive_inference_request(
    request_id: int = Form(...),
    user_id: int = Form(...),
    frame_id: int = Form(...),
    opt_model: str = Form(...),
    est_map: float = Form(...),
    est_time: float = Form(...),
    file: UploadFile = File(...)
):
    """
    The main API endpoint. Clients POST their images and form data here.
    """
    arrival_time = time.time()
    
    # 1. Save the raw image bytes to disk so YOLO can read it later
    file_location = UPLOAD_DIR / f"req{request_id}_u{user_id}_f{frame_id}_{file.filename}"
    with open(file_location, "wb+") as f:
        f.write(await file.read())

    # 2. Wrap all the data into our dataclass
    req = InferenceRequest(
        request_id=request_id,
        user_id=user_id,
        frame_id=frame_id,
        frame_path=str(file_location),
        arrival_time=arrival_time,
        opt_model=opt_model,
        est_map=est_map,
        est_time=est_time
    )

    # 3. Create the Future
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending_responses[request_id] = future

    # 4. Push the request to the queue for the background worker to find
    global_queue.push(req, arrival_time)

    # 5. Suspend this HTTP connection
    try:
        # The code stops executing here and waits until the background worker
        # calls `future.set_result()`
        result = await future 
        return JSONResponse(content=result)
    except asyncio.CancelledError:
        # If the client gives up and disconnects while waiting in the queue,
        # clean up the dictionary to prevent memory leaks.
        pending_responses.pop(request_id, None)
        return JSONResponse(status_code=499, content={"error": "Client Closed Request"})


if __name__ == "__main__":
    # Start the server on port 8000, accepting traffic from any IP
    uvicorn.run(app, host="0.0.0.0", port=8000)