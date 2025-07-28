import time
import subprocess

while True:
    print(f"[{time.ctime()}] Running ckpt_upload.sh...")
    subprocess.run(["bash", "/scratch/nishang/rStar-RL/verl/scripts/azcopy_script/ckpt_upload.sh"])
    print(f"[{time.ctime()}] Done. Sleeping for 1 hour.\n")
    time.sleep(60 * 60)  # 1 hour