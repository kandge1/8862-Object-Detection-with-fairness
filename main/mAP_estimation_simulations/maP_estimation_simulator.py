"""
CARLA Circular Driving Simulation with Multi-Speed Sensor Data Capture
======================================================================
Simulates a single vehicle driving in a circle at multiple max speeds:
5, 10, 20, 30, and 40 m/s

Captures:
  - RGB Camera
  - Depth Camera
  - Semantic Segmentation Camera
  - LiDAR
  - GNSS (GPS)
  - IMU
  - Collision Detector
  - Lane Invasion Detector

Requirements:
  pip install carla numpy opencv-python

Usage:
  python carla_circle_simulation.py [--host HOST] [--port PORT]
"""

import carla
import argparse
import math
import time
import os
import json
import queue
import numpy as np
import cv2

# ─── Configuration ────────────────────────────────────────────────────────────

MAX_SPEEDS = [5, 10, 20, 30, 40]   # m/s
CIRCLE_RADIUS = 30.0               # metres
LAPS_PER_SPEED = 2                 # how many full circles per speed
TICK_INTERVAL = 0.05               # seconds between physics ticks (20 Hz)
OUTPUT_ROOT = "carla_sim_output"

# Sensor specs
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FOV    = 90

LIDAR_CHANNELS    = 64
LIDAR_RANGE       = 50.0
LIDAR_PPS         = 100_000        # points per second
LIDAR_ROTATION_HZ = 20.0

# ─── Helpers ──────────────────────────────────────────────────────────────────

def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def vec3(v) -> dict:
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def save_image(image, path: str):
    """Save a CARLA image (BGRA raw) as PNG."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    cv2.imwrite(path, array[:, :, :3])   # drop alpha


def save_depth(image, path: str):
    """Save a CARLA depth image as a 16-bit PNG (millimetres)."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4)).astype(np.float32)
    # CARLA depth encoding: R + G*256 + B*256^2 metres (normalised to 1000 m)
    depth_m = (array[:, :, 0]
               + array[:, :, 1] * 256.0
               + array[:, :, 2] * 256.0 * 256.0) / (256.0 ** 3 - 1) * 1000.0
    depth_mm = (depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_mm)


def save_lidar(point_cloud, path: str):
    """Save LiDAR point cloud as a binary .npy file (N x 4: x, y, z, intensity)."""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    data = data.reshape((-1, 4))
    np.save(path, data)


# ─── Sensor Manager ───────────────────────────────────────────────────────────

class SensorManager:
    """Attaches sensors to an actor and routes data to per-speed directories."""

    def __init__(self, world: carla.World, vehicle: carla.Actor,
                 blueprint_library, speed_dir: str):
        self.vehicle    = vehicle
        self.speed_dir  = speed_dir
        self.sensors    = []
        self.frame_idx  = 0

        self._event_log = []   # collisions / lane invasions

        bp = blueprint_library

        # ── RGB Camera ──
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        cam_bp.set_attribute("fov", str(CAM_FOV))
        cam_bp.set_attribute("sensor_tick", str(TICK_INTERVAL))
        self._spawn_sensor(world, cam_bp, carla.Transform(
            carla.Location(x=1.5, z=2.4)), self._on_rgb)

        # ── Depth Camera ──
        dep_bp = bp.find("sensor.camera.depth")
        dep_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        dep_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        dep_bp.set_attribute("fov", str(CAM_FOV))
        dep_bp.set_attribute("sensor_tick", str(TICK_INTERVAL))
        self._spawn_sensor(world, dep_bp, carla.Transform(
            carla.Location(x=1.5, z=2.4)), self._on_depth)

        # ── Semantic Segmentation Camera ──
        seg_bp = bp.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        seg_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        seg_bp.set_attribute("fov", str(CAM_FOV))
        seg_bp.set_attribute("sensor_tick", str(TICK_INTERVAL))
        self._spawn_sensor(world, seg_bp, carla.Transform(
            carla.Location(x=1.5, z=2.4)), self._on_seg)

        # ── LiDAR ──
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels",          str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("range",             str(LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second", str(LIDAR_PPS))
        lidar_bp.set_attribute("rotation_frequency",str(LIDAR_ROTATION_HZ))
        lidar_bp.set_attribute("sensor_tick",       str(TICK_INTERVAL))
        self._spawn_sensor(world, lidar_bp, carla.Transform(
            carla.Location(z=2.5)), self._on_lidar)

        # ── GNSS ──
        gnss_bp = bp.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", str(TICK_INTERVAL))
        self._spawn_sensor(world, gnss_bp, carla.Transform(), self._on_gnss)
        self._gnss_log = []

        # ── IMU ──
        imu_bp = bp.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(TICK_INTERVAL))
        self._spawn_sensor(world, imu_bp, carla.Transform(), self._on_imu)
        self._imu_log = []

        # ── Collision ──
        col_bp = bp.find("sensor.other.collision")
        self._spawn_sensor(world, col_bp, carla.Transform(), self._on_collision)

        # ── Lane Invasion ──
        lane_bp = bp.find("sensor.other.lane_invasion")
        self._spawn_sensor(world, lane_bp, carla.Transform(), self._on_lane)

        # Create output sub-directories
        for d in ["rgb", "depth", "seg", "lidar"]:
            mkdir(os.path.join(speed_dir, d))

    # ── internal helpers ──

    def _spawn_sensor(self, world, bp, transform, callback):
        sensor = world.spawn_actor(bp, transform, attach_to=self.vehicle)
        sensor.listen(callback)
        self.sensors.append(sensor)

    def _frame_str(self):
        return f"{self.frame_idx:06d}"

    # ── callbacks ──

    def _on_rgb(self, image):
        path = os.path.join(self.speed_dir, "rgb", f"{self._frame_str()}.png")
        save_image(image, path)
        self.frame_idx += 1

    def _on_depth(self, image):
        path = os.path.join(self.speed_dir, "depth", f"{self._frame_str()}.png")
        save_depth(image, path)

    def _on_seg(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        path = os.path.join(self.speed_dir, "seg", f"{self._frame_str()}.png")
        save_image(image, path)

    def _on_lidar(self, point_cloud):
        path = os.path.join(self.speed_dir, "lidar", f"{self._frame_str()}.npy")
        save_lidar(point_cloud, path)

    def _on_gnss(self, gnss):
        self._gnss_log.append({
            "frame":     gnss.frame,
            "timestamp": gnss.timestamp,
            "lat":       gnss.latitude,
            "lon":       gnss.longitude,
            "alt":       gnss.altitude,
        })

    def _on_imu(self, imu):
        self._imu_log.append({
            "frame":         imu.frame,
            "timestamp":     imu.timestamp,
            "accel":         vec3(imu.accelerometer),
            "gyro":          vec3(imu.gyroscope),
            "compass_deg":   math.degrees(imu.compass),
        })

    def _on_collision(self, event):
        actor_type = event.other_actor.type_id if event.other_actor else "unknown"
        self._event_log.append({
            "type":      "collision",
            "frame":     event.frame,
            "timestamp": event.timestamp,
            "other":     actor_type,
            "impulse":   vec3(event.normal_impulse),
        })
        print(f"  [!] Collision with {actor_type} at frame {event.frame}")

    def _on_lane(self, event):
        markings = [str(m) for m in event.crossed_lane_markings]
        self._event_log.append({
            "type":      "lane_invasion",
            "frame":     event.frame,
            "timestamp": event.timestamp,
            "markings":  markings,
        })

    # ── public ──

    def flush_logs(self):
        """Write accumulated GNSS / IMU / event logs to JSON files."""
        for name, data in [("gnss", self._gnss_log),
                           ("imu",  self._imu_log),
                           ("events", self._event_log)]:
            path = os.path.join(self.speed_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        print(f"  Logs written to {self.speed_dir}/")

    def destroy(self):
        for s in self.sensors:
            if s.is_alive:
                s.stop()
                s.destroy()
        self.sensors.clear()


# ─── Circular Controller ──────────────────────────────────────────────────────

class CircleController:
    """
    Simple pure-pursuit-style controller that keeps the vehicle on a
    horizontal circle of radius R centred at `centre`.
    """

    def __init__(self, centre: carla.Location, radius: float, max_speed: float):
        self.cx        = centre.x
        self.cy        = centre.y
        self.R         = radius
        self.max_speed = max_speed   # m/s

    def run_step(self, vehicle: carla.Vehicle) -> carla.VehicleControl:
        t   = vehicle.get_transform()
        v   = vehicle.get_velocity()
        loc = t.location

        # Vector from centre to vehicle
        dx = loc.x - self.cx
        dy = loc.y - self.cy
        dist = math.sqrt(dx * dx + dy * dy) or 0.001

        # Target point slightly ahead on the circle (look-ahead ≈ 5 m of arc)
        angle_now    = math.atan2(dy, dx)
        look_ahead   = 5.0 / self.R             # radians
        angle_target = angle_now + look_ahead   # go CCW

        tx = self.cx + self.R * math.cos(angle_target)
        ty = self.cy + self.R * math.sin(angle_target)

        # Direction from vehicle to target
        tdx = tx - loc.x
        tdy = ty - loc.y
        target_yaw = math.degrees(math.atan2(tdy, tdx))

        # Heading error
        yaw_err = target_yaw - t.rotation.yaw
        yaw_err = (yaw_err + 180) % 360 - 180   # normalise to [-180, 180]

        # Steer proportionally (clamp to [-1, 1])
        steer = max(-1.0, min(1.0, yaw_err / 45.0))

        # Speed control
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        throttle = max(0.0, min(1.0, (self.max_speed - speed) / self.max_speed))
        brake    = 0.0
        if speed > self.max_speed + 0.5:
            throttle = 0.0
            brake    = 0.3

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
        )


# ─── Simulation Runner ────────────────────────────────────────────────────────

def run_speed(world, vehicle, blueprint_library, max_speed: float):
    print(f"\n{'='*60}")
    print(f"  Speed: {max_speed} m/s  |  Laps: {LAPS_PER_SPEED}  |  Radius: {CIRCLE_RADIUS} m")
    print(f"{'='*60}")

    speed_dir = mkdir(os.path.join(OUTPUT_ROOT, f"speed_{int(max_speed):02d}mps"))
    sensors   = SensorManager(world, vehicle, blueprint_library, speed_dir)

    # Place vehicle at the circle's edge, facing the tangent direction
    start_loc  = vehicle.get_transform().location
    centre     = carla.Location(x=start_loc.x, y=start_loc.y + CIRCLE_RADIUS, z=start_loc.z)
    controller = CircleController(centre, CIRCLE_RADIUS, max_speed)

    # Compute how long one lap takes (circumference / speed), with margin
    lap_time      = (2 * math.pi * CIRCLE_RADIUS) / max_speed
    total_time    = lap_time * LAPS_PER_SPEED * 1.1  # 10 % headroom
    start_time    = time.time()
    frame_count   = 0

    # Teleport to circle start
    spawn_tf = carla.Transform(
        carla.Location(x=centre.x + CIRCLE_RADIUS, y=centre.y, z=start_loc.z + 0.5),
        carla.Rotation(yaw=90)   # facing +Y tangent
    )
    vehicle.set_transform(spawn_tf)
    vehicle.set_target_velocity(carla.Vector3D(0, max_speed * 0.5, 0))
    time.sleep(0.5)

    print(f"  Running for ~{total_time:.1f} s …")

    while time.time() - start_time < total_time:
        control = controller.run_step(vehicle)
        vehicle.apply_control(control)
        world.tick()
        frame_count += 1

        # Progress indicator every 5 s
        elapsed = time.time() - start_time
        if frame_count % 100 == 0:
            v = vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            print(f"    t={elapsed:.1f}s  speed={spd:.2f} m/s  frames={frame_count}")

    # Stop the vehicle
    vehicle.apply_control(carla.VehicleControl(brake=1.0))
    world.tick()

    sensors.flush_logs()
    sensors.destroy()

    # Per-speed metadata
    meta = {
        "max_speed_mps":   max_speed,
        "laps":            LAPS_PER_SPEED,
        "radius_m":        CIRCLE_RADIUS,
        "total_frames":    frame_count,
        "elapsed_s":       time.time() - start_time,
        "camera_res":      f"{CAM_WIDTH}x{CAM_HEIGHT}",
        "lidar_channels":  LIDAR_CHANNELS,
        "lidar_range_m":   LIDAR_RANGE,
    }
    with open(os.path.join(speed_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done – {frame_count} frames saved to {speed_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CARLA circular drive simulation")
    parser.add_argument("--host",    default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port",    default=2000,        type=int, help="CARLA server port")
    parser.add_argument("--tm-port", default=8000,        type=int, help="Traffic Manager port")
    parser.add_argument("--map",     default=None,        help="Map name (default: server default)")
    parser.add_argument("--speeds",  default=None, nargs="+", type=float,
                        help="Override speed list, e.g. --speeds 5 10 20")
    args = parser.parse_args()

    speeds = args.speeds or MAX_SPEEDS

    print(f"Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    if args.map:
        world = client.load_world(args.map)

    # Synchronous mode for deterministic simulation
    settings = world.get_settings()
    settings.synchronous_mode      = True
    settings.fixed_delta_seconds   = TICK_INTERVAL
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Pick a vehicle blueprint
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

    # Find a suitable spawn point (flat area)
    spawn_points = world.get_map().get_spawn_points()
    spawn_tf     = spawn_points[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    vehicle.set_autopilot(False)

    print(f"Spawned vehicle: {vehicle.type_id} at {spawn_tf.location}")
    print(f"Output directory: {os.path.abspath(OUTPUT_ROOT)}")
    print(f"Speeds to simulate: {speeds} m/s")

    try:
        for spd in speeds:
            run_speed(world, vehicle, blueprint_library, spd)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("\nCleaning up …")
        vehicle.destroy()

        # Restore async mode
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("Done.")


if __name__ == "__main__":
    main()