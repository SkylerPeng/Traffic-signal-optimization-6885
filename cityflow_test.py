import cityflow
import os

config_path = "examples/config.json"

if not os.path.exists(config_path):
    print("Cannot find examples/config.json, please confirm that you are under CityFlow!")
    exit()

print("Initiating CityFlow engine...")
eng = cityflow.Engine(config_path, thread_num=1)

print("Running simulization (100 step)...")
for step in range(100):
    eng.next_step()

    # Print the number of cars every 10 steps
    if step % 10 == 0:
        count = eng.get_vehicle_count()
        print(f"   Step {step}: Roadnet has {count} cars")

print("\n Validation successfully!")