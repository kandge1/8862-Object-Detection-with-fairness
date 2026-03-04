import carla
import time
import random

# Fixed seed for deterministic spawns: same vehicle types at same spots every run
SPAWN_SEED = 67
random.seed(SPAWN_SEED)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Despawn all existing vehicles so we start from a clean slate
for actor in world.get_actors():
    if actor.type_id.startswith("vehicle."):
        try:
            actor.destroy()
        except RuntimeError:
            pass
# Brief pause so destroy commands are applied before we spawn
time.sleep(0.5)

blueprint_library = world.get_blueprint_library()
# Spawn points are fixed per map: same order every time (road positions from the map)
spawn_points = world.get_map().get_spawn_points()
num_vehicles = min(32, len(spawn_points))

# Spawn ego vehicle (Tesla with camera and LiDAR) at first spawn point
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
ego_vehicle.set_autopilot(True)

# Spawn remaining vehicles at fixed spawn_points; with SPAWN_SEED, same blueprint at same index every run
vehicle_blueprints = list(blueprint_library.filter('vehicle.*'))
all_vehicles = [ego_vehicle]
for i in range(1, num_vehicles):
    bp = random.choice(vehicle_blueprints)
    sp = spawn_points[i]
    try:
        v = world.spawn_actor(bp, sp)
        v.set_autopilot(True)
        all_vehicles.append(v)
    except RuntimeError:
        pass

# Attach camera to ego vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
camera.listen(lambda image: image.save_to_disk(f'output/{image.frame}.png'))

# Attach LiDAR to ego vehicle
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.5)), attach_to=ego_vehicle)
lidar.listen(lambda data: data.save_to_disk(f'output/lidar/{data.frame}.ply'))

# Run for 60 seconds, drawing "ego" marker above the ego vehicle each frame
duration = 300.0
start = time.time()
while time.time() - start < duration:
    t = ego_vehicle.get_transform()
    marker_location = carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + 3.5)
    world.debug.draw_string(marker_location, "ego", draw_shadow=False, color=carla.Color(255, 200, 0), life_time=0.15)
    time.sleep(0.1)

# Cleanup
camera.stop()
camera.destroy()
lidar.destroy()
for v in all_vehicles:
    try:
        v.destroy()
    except RuntimeError:
        pass