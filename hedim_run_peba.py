import subprocess
import csv
import re
import os
import argparse

def run_hedim(
    data_file_path: str,
    n: int,
    sample_id_a: int,
    sample_id_b: int,
    executable_path: str = 'peba1/build/peba1'
):
    # Run the executable
    cmd = [executable_path, data_file_path, str(n), str(sample_id_a), str(sample_id_b)]
    print("[hedim_run_peba.py] Calling " + " ".join(cmd))
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[hedim_run_peba.py] Error running executable: {e}")
        print(f"[hedim_run_peba.py] stderr:\n{e.stderr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PEBA binary")
    parser.add_argument('data_file_path', help='Path to the .txt vector file')
    parser.add_argument('n', type=int, help='Vector length (nslots)')
    parser.add_argument('sample_id_a', type=int, help='First sample ID')
    parser.add_argument('sample_id_b', type=int, help='Second sample ID')
    parser.add_argument('--executable_path', default='peba1/build/peba1', help='Path to the compiled PEBA executable')

    args = parser.parse_args()
    run_hedim(
        args.data_file_path,
        args.n,
        args.sample_id_a,
        args.sample_id_b,
        args.executable_path
    )