import subprocess

from core.env_loader import init_env


init_env()
model_name = ""
result = subprocess.run(["accelerate", "estimate-memory", model_name, "--library_name", "transformers"])
if result.returncode == 0:
    print(result.stdout)
else:
    print("Error:", result.stderr)
