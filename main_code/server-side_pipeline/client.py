#Tracking, without estimator
import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import requests


# Client-side lookup table. In the current joint test, this is only used to fill
# metadata expected by the existing server. The server may ignore it if fixed to yolo26n.
# run: python3 client_v2.py --frames-folder frames/img1

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


@dataclass
class ClientState:
    user_id: int
    current_frame_id: int
    tracking_age: int
    last_offload_frame_id: int
    recent_failures: int = 0


@dataclass
class ScheduleDecision:
    model_name: str
    effective_map: float = 0.0

    
class DynamicRomaScheduler:
    """
    ROMA-inspired dynamic scheduler. 
    Selects the optimal model by penalizing base mAP based on estimated latency 
    and recent object tracking stability (IoU loss).
    """
    def __init__(self, map_table: Dict[str, Dict[str, float]], target_fps: int = 30) -> None:
        self.map_table = map_table
        self.target_fps = target_fps
        # Base accuracy retained per dropped frame (1.0 = perfect retention, 0.0 = total loss)
        self.base_retention = 0.98 


    def decide(self, state: ClientState) -> ScheduleDecision:
        best_model = "yolov8n_640"
        best_effective_map = -1.0

        dynamic_retention = self.base_retention - (state.recent_failures * 0.02)
        dynamic_retention = max(0.5, dynamic_retention)

        for model_name, stats in self.map_table.items():
            base_map = stats["map"]
            latency_sec = stats["time"] / 1000.0 
            # ROMA Eq 12: Estimate frame block size (how many frames we drop waiting for inference)
            frames_dropped = int(self.target_fps * latency_sec) + 1
            # ROMA Eq 16 & 17: Penalize the AP based on how many frames we drop
            effective_map = base_map * (dynamic_retention ** frames_dropped)

            if effective_map > best_effective_map:
                best_effective_map = effective_map
                best_model = model_name

        return ScheduleDecision(model_name=best_model, effective_map=best_effective_map)


class KCFTrackerManager:
    def __init__(self) -> None:
        self.tracked_objects: List[Dict[str, Any]] = []
        self.last_server_detections: List[Dict[str, Any]] = []

    @staticmethod
    def _create_tracker():
        # OpenCV has moved tracker constructors across versions.
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        raise RuntimeError(
            "KCF tracker is unavailable. Install opencv-contrib-python, e.g.\n"
            "pip install opencv-contrib-python"
        )

    def reset_from_detections(self, frame, detections: List[Dict[str, Any]]) -> int:
        self.tracked_objects = []
        self.last_server_detections = detections or []

        height, width = frame.shape[:2]
        for det in self.last_server_detections:
            bbox = det.get("bbox_xywh_int")

            if bbox and len(bbox) == 4:
                x, y, w, h = [int(v) for v in bbox]
            else:
                xyxy = det.get("bbox_xyxy")
                if not xyxy or len(xyxy) != 4:
                    continue

                x1, y1, x2, y2 = [float(v) for v in xyxy]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                x = int(round(x1))
                y = int(round(y1))
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))

            x = max(0, min(width - 1, int(x)))
            y = max(0, min(height - 1, int(y)))
            w = max(1, min(width - x, int(w)))
            h = max(1, min(height - y, int(h)))

            bbox_tuple = (x, y, w, h)

            tracker = self._create_tracker()
            ok = tracker.init(frame, bbox_tuple)

            if ok is False:
                continue

            self.tracked_objects.append({
                "tracker": tracker,
                "class_name": det.get("class_name", "object"),
                "confidence": float(det.get("confidence", 0.0)),
                "bbox_xywh": bbox_tuple,
            })

        return len(self.tracked_objects)

    def update(self, frame) -> Tuple[List[Dict[str, Any]], int]:
        active: List[Dict[str, Any]] = []
        failures = 0

        for obj in self.tracked_objects:
            ok, bbox = obj["tracker"].update(frame)
            if ok:
                x, y, w, h = [float(v) for v in bbox]
                obj["bbox_xywh"] = (x, y, w, h)
                active.append(obj)
            else:
                failures += 1

        self.tracked_objects = active
        return active, failures

    def has_active_tracker(self) -> bool:
        return len(self.tracked_objects) > 0

    def draw_tracking(self, frame) -> None:
        for obj in self.tracked_objects:
            x, y, w, h = [int(v) for v in obj["bbox_xywh"]]
            label = f"KCF {obj.get('class_name', 'object')}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_last_server_detections(self, frame) -> None:
        for det in self.last_server_detections:
            xyxy = det.get("bbox_xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
          

        return len(self.tracked_objects)

    def update(self, frame) -> Tuple[List[Dict[str, Any]], int]:
        active: List[Dict[str, Any]] = []
        failures = 0

        for obj in self.tracked_objects:
            ok, bbox = obj["tracker"].update(frame)
            if ok:
                x, y, w, h = [float(v) for v in bbox]
                obj["bbox_xywh"] = (x, y, w, h)
                active.append(obj)
            else:
                failures += 1

        self.tracked_objects = active
        return active, failures

    def has_active_tracker(self) -> bool:
        return len(self.tracked_objects) > 0

    def draw_tracking(self, frame) -> None:
        for obj in self.tracked_objects:
            x, y, w, h = [int(v) for v in obj["bbox_xywh"]]
            label = f"KCF {obj.get('class_name', 'object')}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_last_server_detections(self, frame) -> None:
        for det in self.last_server_detections:
            xyxy = det.get("bbox_xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
            label = f"SERVER {det.get('class_name', 'object')} {float(det.get('confidence', 0.0)):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


class EdgeClientJointTest:
    def __init__(
        self,
        user_id: int,
        frames_folder: str,
        server_url: str,
        target_fps: int = 30,
        fixed_model: str = "yolov8n_640",
        max_tracking_age: int = 10,
        visualize: bool = True,
        log_path: str = "client_joint_log.jsonl",
    ) -> None:
        self.user_id = user_id
        self.server_url = server_url
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.max_tracking_age = max_tracking_age
        self.visualize = visualize
        self.log_path = Path(log_path)

        self.frame_paths = self._load_frame_paths(frames_folder)
        self.total_frames = len(self.frame_paths)

        self.state = ClientState(
            user_id=user_id,
            current_frame_id=0,
            tracking_age=0,
            last_offload_frame_id=-1,
        )
        self.scheduler = DynamicRomaScheduler(map_table=CLIENT_MAP_TABLE, target_fps=target_fps)
        self.tracker_manager = KCFTrackerManager()

        self.next_request_id = 0
        self.offload_in_flight = False
        self.lock = threading.Lock()
        self.latest_frame_for_tracker: Optional[Any] = None

        self.num_offloads = 0
        self.num_server_success = 0
        self.num_tracking_frames = 0
        self.num_tracking_failures = 0

        self.sum_map = 0.0
        self.sum_map_square = 0.0
        self.frame_count_for_map = 0

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    @staticmethod
    def _load_frame_paths(folder: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[WARN] Folder does not exist: {folder}")
            return []
        return sorted(str(p) for p in folder_path.iterdir() if p.suffix.lower() in exts)

    def _append_log(self, record: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(self) -> None:
        if not self.frame_paths:
            print(f"Client {self.user_id}: no frames found. Exiting.")
            return

        print(f"[START] user={self.user_id}, frames={self.total_frames}, fps={self.target_fps}")
        print(f"[SERVER] {self.server_url}")
        print(f"[LOG] {self.log_path}")
        print("[VIEW] green=KCF tracking, red=latest server detection, blue text=status")

        for frame_path in self.frame_paths:
            loop_start = time.time()
            frame_id = self.state.current_frame_id
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARN] Failed to read frame: {frame_path}")
                self.state.current_frame_id += 1
                continue

            with self.lock:
                self.latest_frame_for_tracker = frame.copy()

            active_tracks, failures = self.tracker_manager.update(frame)
            self.num_tracking_failures += failures
            if active_tracks:
                self.num_tracking_frames += 1

            self.state.recent_failures = failures
            decision = self.scheduler.decide(self.state)

            self.frame_count_for_map += 1
            self.sum_map += decision.effective_map
            avg_map = self.sum_map / self.frame_count_for_map

            # --- Calculate Jain's Index for active objects in THIS frame ---
            if active_tracks:
                sum_conf = sum(obj.get("confidence", 0.0) for obj in active_tracks)
                sum_sq_conf = sum(obj.get("confidence", 0.0) ** 2 for obj in active_tracks)
                n_objs = len(active_tracks)
                obj_jains_index = (sum_conf ** 2) / (n_objs * sum_sq_conf) if sum_sq_conf > 0 else 1.0
            else:
                obj_jains_index = 1.0  # Default to 1.0 if no objects are on screen

            should_offload = self._should_offload()
            event = "track"

            if should_offload:
                event = "offload"
                self._start_offload_thread(frame_path, decision.model_name, frame_id)
                self.state.last_offload_frame_id = frame_id
                self.state.tracking_age = 0
            else:
                self.state.tracking_age += 1

            self._append_log({
                "type": "client_frame",
                "timestamp": time.time(),
                "user_id": self.user_id,
                "frame_id": frame_id,
                "event": event,
                "tracking_age": self.state.tracking_age,
                "active_trackers": len(active_tracks),
                "tracking_failures_this_frame": failures,
                "offload_in_flight": self.offload_in_flight,
            })


            self._print_frame_status(frame_id, event, active_tracks, failures, decision, avg_map, obj_jains_index)

            if self.visualize:
                display = frame.copy()
                self.tracker_manager.draw_last_server_detections(display)
                self.tracker_manager.draw_tracking(display)
                self._draw_status(display, frame_id, event, failures)
                cv2.imshow(f"Client {self.user_id} joint test", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.state.current_frame_id += 1
            elapsed = time.time() - loop_start
            if elapsed < self.frame_delay:
                time.sleep(self.frame_delay - elapsed)

        while self.offload_in_flight:
            time.sleep(0.05)

        if self.visualize:
            cv2.destroyAllWindows()

        print("[DONE]")
        print(f"  offloads sent: {self.num_offloads}")
        print(f"  server successes: {self.num_server_success}")
        print(f"  frames with active tracking: {self.num_tracking_frames}")
        print(f"  tracking failures: {self.num_tracking_failures}")

    def _should_offload(self) -> bool:
        with self.lock:
            in_flight = self.offload_in_flight
        if in_flight:
            return False
        if not self.tracker_manager.has_active_tracker():
            return True
        if self.state.tracking_age >= self.max_tracking_age:
            return True
        return False

    def _start_offload_thread(self, frame_path: str, requested_model: str, frame_id: int) -> None:
        with self.lock:
            self.offload_in_flight = True
            request_id = self.next_request_id
            self.next_request_id += 1
            self.num_offloads += 1

        thread = threading.Thread(
            target=self._send_inference_request,
            args=(request_id, frame_path, requested_model, frame_id),
            daemon=True,
        )
        thread.start()

    def _send_inference_request(self, request_id: int, frame_path: str, requested_model: str, captured_frame_id: int) -> None:
        stats = CLIENT_MAP_TABLE.get(requested_model, {"map": 0.5, "time": 5.0})
        payload = {
            "request_id": request_id,
            "user_id": self.user_id,
            "frame_id": captured_frame_id,
            "opt_model": requested_model,
            "est_map": stats["map"],
            "est_time": stats["time"],
            # Extra state fields are harmless for a strict FastAPI endpoint if omitted there;
            # requests will still send them as form fields. Existing server can ignore them.
            "tracking_age": self.state.tracking_age,
            "last_offload_frame_id": self.state.last_offload_frame_id,
        }

        print(f"[SEND] req={request_id}, frame={captured_frame_id}, model_hint={requested_model}, file={Path(frame_path).name}")
        start = time.time()
        try:
            with open(frame_path, "rb") as f:
                response = requests.post(self.server_url, data=payload, files={"file": f}, timeout=60)
            rtt = time.time() - start

            if response.status_code != 200:
                print(f"[SERVER-ERR] req={request_id}, status={response.status_code}, body={response.text[:300]}")
                self._append_log({
                    "type": "server_error",
                    "timestamp": time.time(),
                    "request_id": request_id,
                    "frame_id": captured_frame_id,
                    "status_code": response.status_code,
                    "body": response.text[:1000],
                    "rtt_sec": rtt,
                })
                return

            result = response.json()
            detections = result.get("detections", []) or []
            num_detections = result.get("num_detections", len(detections))
            print(f"[RECV] req={request_id}, frame={captured_frame_id}, rtt={rtt:.3f}s, detections={num_detections}")

            # for i, det in enumerate(detections[:10]):
            #     print(
            #         f"       det[{i}] class={det.get('class_name')} "
            #         f"conf={float(det.get('confidence', 0.0)):.3f} "
            #         f"bbox={det.get('bbox_xyxy')}"
            #     )
            # if len(detections) > 10:
            #     print(f"       ... {len(detections) - 10} more detections")

            # Initialize KCF trackers from server detection using the returned frame image.
            # We use the exact uploaded frame to align detection boxes with pixels.
            init_frame = cv2.imread(frame_path)
            tracker_count = 0
            if init_frame is not None:
                with self.lock:
                    tracker_count = self.tracker_manager.reset_from_detections(init_frame, detections)
                    self.state.tracking_age = 0
                    self.num_server_success += 1

            print(f"[TRACKER-INIT] req={request_id}, initialized={tracker_count}")
            self._append_log({
                "type": "server_result",
                "timestamp": time.time(),
                "request_id": request_id,
                "user_id": self.user_id,
                "frame_id": captured_frame_id,
                "rtt_sec": rtt,
                "num_detections": num_detections,
                "initialized_trackers": tracker_count,
                "annotated_image_path_on_server": result.get("annotated_image_path"),
                "detections": detections,
            })

        except requests.exceptions.ConnectionError:
            print("[CONNECTION-ERR] Ensure the server is running and SERVER_URL is correct.")
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] req={request_id}, frame={captured_frame_id}")
        except Exception as exc:
            print(f"[CLIENT-ERR] req={request_id}: {exc}")
        finally:
            with self.lock:
                self.offload_in_flight = False


    def _print_frame_status(self, frame_id: int, event: str, active_tracks: List[Dict[str, Any]], failures: int, decision: ScheduleDecision, avg_map: float, obj_jains_index: float) -> None:
        if event == "offload" or failures > 0 or frame_id % max(1, self.target_fps) == 0:
            print(
                f"[FRAME] {frame_id:05d} event={event:<7} "
                f"model={decision.model_name:<11} "
                f"mAP={decision.effective_map:.3f} Avg_mAP={avg_map:.3f} Obj_Jain={obj_jains_index:.3f} "
                f"failures={failures:<2} in_flight={self.offload_in_flight}"
            )

    def _draw_status(self, frame, frame_id: int, event: str, failures: int) -> None:
        text = (
            f"frame={frame_id} event={event} age={self.state.tracking_age} "
            f"trackers={len(self.tracker_manager.tracked_objects)} failures={failures} "
            f"in_flight={self.offload_in_flight}"
        )
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint client test with HTTP offload + KCF tracking visualization.")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--frames-folder", type=str, default="frames/user1")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000/infer")
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--fixed-model", type=str, default="yolov8n_640")
    parser.add_argument("--max-tracking-age", type=int, default=10)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--log-path", type=str, default="client_joint_log.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = EdgeClientJointTest(
        user_id=args.user_id,
        frames_folder=args.frames_folder,
        server_url=args.server_url,
        target_fps=args.target_fps,
        fixed_model=args.fixed_model,
        max_tracking_age=args.max_tracking_age,
        visualize=not args.no_visualize,
        log_path=args.log_path,
    )
    client.run()
