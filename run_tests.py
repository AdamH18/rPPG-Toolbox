import subprocess
import os

config_file_loc = "/home/adamh/rPPG/rPPG-Toolbox/configs/my_configs"
configs = ["PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml", "PURE_PURE_UBFC-rPPG_PHYSFORMER.yaml"]
results_loc = "/home/adamh/rPPG/rPPG-Toolbox/results"

os.makedirs(results_loc, exist_ok=True)

for config in configs:
    print(f"Starting run for config {config}")
    res_file = os.path.join(results_loc, config.split('.')[0] + ".txt")
    with open(res_file, 'w') as results_file:
        try:
            subprocess.run(["bash", "-i", "/home/adamh/rPPG/rPPG-Toolbox/run_tests.sh", f"{os.path.join(config_file_loc, config)}"], stdout=results_file, stderr=subprocess.STDOUT, check=True)
        except Exception as e:
            print(f"Config {config} errored with error {e}")
        results_file.flush()

print("Done!")
