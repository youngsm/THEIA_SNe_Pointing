import subprocess
from glob2 import glob
import re
import argparse
from sys import argv

def get_gpuid():
    gpu_id = None
    if "--gpu" in argv:
        for i in range(len(argv)):
            if len(argv) > i and argv[i] == "--gpu":
                gpu_id = argv[i+1]
    return gpu_id

def main():

    out_dir = "/home/youngsam/portal/sims/2023-03-21_SNe"
    cmd = "pyrat reconstruct_supernovae.py"
    
    if (gpuid:=get_gpuid()):
        print(f"Using GPU {gpuid}")
        cmd = f"CUDA_VISIBLE_DEVICES={gpuid} {cmd}"

    def get_run():
        runmax = 0
        for file in glob(f"{out_dir}/unpack_*.h5"):
            search = re.search(r"unpack_(\d+).h5", file)
            if search:
                runmax = max(runmax, int(search.group(1)))
        run = runmax+1
        return run

    while (run:=get_run()) < 500:
        print(f"Running SNe simulation -- Run {run}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
