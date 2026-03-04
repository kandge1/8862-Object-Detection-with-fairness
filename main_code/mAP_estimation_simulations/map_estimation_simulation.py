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
num_vehicles = min(64, len(spawn_points))

# Spawn vehicles at fixed spawn_points; with SPAWN_SEED, same blueprint at same index every run
vehicle_blueprints = list(blueprint_library.filter('vehicle.*'))
all_vehicles = []
for i in range(num_vehicles):
    bp = random.choice(vehicle_blueprints)
    sp = spawn_points[i]
    try:
        v = world.spawn_actor(bp, sp)
        v.set_autopilot(True)
        all_vehicles.append(v)
    except RuntimeError:
        pass

# Run for 300 seconds (no ego, no sensors—just vehicles driving)
duration = 300.0
start = time.time()
try:
    while time.time() - start < duration:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    print("Cleaning up vehicles...")
    # Destroy all vehicles in the world (robust on Ctrl+C; doesn't rely on all_vehicles list)
    for actor in world.get_actors():
        if actor.type_id.startswith("vehicle."):
            try:
                actor.destroy()
            except RuntimeError:
                pass
    print("Done.")