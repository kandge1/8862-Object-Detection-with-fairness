import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import requests


# 1. Client mAP Table
CLIENT_MAP_TABLE: Dict[str, Dict[str, float]] = {
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

# 2. CLIENT STATE & SCHEDULING
@dataclass
class ClientState:
    user_id: int
    current_frame_id: int
    tracking_age: int
    last_offload_frame_id: int

@dataclass
class ScheduleDecision:
    model_name: str

class EdgeScheduler:
    def decide(self, state: ClientState) -> ScheduleDecision:
        # The scheduler now decides model
        # add code
        return ScheduleDecision(
            model_name="yolov8n_640" 
        )

# 3. EDGE CLIENT APPLICATION
class EdgeClient:
    def __init__(self, user_id: int, frames_folder: str, server_url: str, target_fps: int = 30):
        self.user_id = user_id
        self.server_url = server_url
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        self.frame_paths = self._load_frame_paths(frames_folder)
        self.total_frames = len(self.frame_paths)
        
        self.state = ClientState(
            user_id=self.user_id,
            current_frame_id=0,
            tracking_age=0,
            last_offload_frame_id=-1
        )
        self.scheduler = EdgeScheduler()
        self.next_request_id = 0
        
        # Network state flag
        self.offload_in_flight = False

    def _load_frame_paths(self, folder: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder '{folder}' does not exist.")
            return []
        return sorted(str(p) for p in folder_path.iterdir() if p.suffix.lower() in exts)

    def run(self):
        if not self.frame_paths:
            print(f"Client {self.user_id}: No frames to process. Exiting.")
            return

        print(f"Client {self.user_id} starting at {self.target_fps} FPS. Wait-for-response offloading enabled...")

        for frame_path in self.frame_paths:
            loop_start_time = time.time()

            # 1. Ask the local scheduler for the model decision
            decision = self.scheduler.decide(self.state)
            
            # 2. Dynamic Offload Trigger: Only offload if the network is free!
            should_offload = not self.offload_in_flight

            if should_offload:
                # Lock the network
                self.offload_in_flight = True
                
                # Launch the network request in a background thread so the camera loop doesn't freeze
                threading.Thread(
                    target=self._send_inference_request,
                    args=(frame_path, decision.model_name, self.state.current_frame_id),
                    daemon=True # Daemon threads die automatically if the main script finishes
                ).start()
                
                self.state.last_offload_frame_id = self.state.current_frame_id
                self.state.tracking_age = 0
            else:
                self.state.tracking_age += 1

            # Advance frame counter
            self.state.current_frame_id += 1

            # Enforce camera FPS pacing
            elapsed = time.time() - loop_start_time
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)

        # Wait a moment at the end for the final in-flight frame to return
        while self.offload_in_flight:
            time.sleep(0.1)
        print("Video stream finished.")

    def _send_inference_request(self, frame_path: str, requested_model: str, captured_frame_id: int):
        """Runs in a background thread."""
        print(f"[Client {self.user_id}] Offloading Frame {captured_frame_id} (Model: {requested_model})...")
        
        stats = CLIENT_MAP_TABLE.get(requested_model, {"map": 0.5, "time": 5.0})
        
        payload = {
            "request_id": self.next_request_id,
            "user_id": self.state.user_id,
            "frame_id": captured_frame_id, # Use the ID captured at thread launch
            "opt_model": requested_model,
            "est_map": stats["map"],
            "est_time": stats["time"]
        }
        self.next_request_id += 1

        try:
            with open(frame_path, "rb") as f:
                response = requests.post(self.server_url, data=payload, files={"file": f})
                
            if response.status_code == 200:
                result_data = response.json()
                print(f"  -> Returned Frame {captured_frame_id}: Found {result_data.get('num_detections')} objects.")
            else:
                print(f"  -> Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print("  -> Connection Error: Ensure the server is running.")
            
        finally:
            # CRITICAL: Unlock the network when the response arrives (or fails)
            # This triggers the main loop to offload the very next frame it sees.
            self.offload_in_flight = False

if __name__ == "__main__":
    # same server
    SERVER_URL = "http://127.0.0.1:8000/infer"

    # # different servers
    # SERVER_URL = "http://10.42.0.5:8000/infer"
    
    client = EdgeClient(
        user_id=1, 
        frames_folder="frames/user1", 
        server_url=SERVER_URL, 
        target_fps=30
    )
    
    client.run()