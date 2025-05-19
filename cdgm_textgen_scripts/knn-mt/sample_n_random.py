import sys
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--prefix", type=str, help="Prefix for the data files")
argparser.add_argument("--out", type=str, help="Output file prefix")
argparser.add_argument("--ntarget", type=int, default=1000, help="Number of target samples to generate")

args = argparser.parse_args()
src='de'
tgt='en'
prefix = sys.argv[1]
ntotal = 2000
ntarget = int(sys.argv[2])

random_samples = np.sort(np.random.randint(0, ntotal, ntarget))
processed = 0
buffer_processed_src = []
buffer_processed_tgt = []


with open(f"{prefix}.{src}", 'r') as nsrc, open(f"{prefix}.{tgt}", 'r') as ntgt:
    lines_src=nsrc.readlines()
    lines_tgt = ntgt.readlines()
    for i in random_samples:
        buffer_processed_src.append(lines_src[i])
        buffer_processed_tgt.append(lines_tgt[i])
        processed += 1
        if processed % 100000 == 0:
            with open(f"{prefix}.rand{ntarget}.{src}", 'a') as csrc, open(f"{prefix}.rand{ntarget}.{tgt}", 'a') as ctgt:
                for line1 in buffer_processed_src:
                    csrc.write(f"{line1}\n")
                for line2 in buffer_processed_tgt:
                    ctgt.write(f"{line2}\n")
            buffer_processed_src = []
            buffer_processed_tgt = []

            print(f"Processed {processed} lines")

    with open(f"{prefix}.rand{ntarget}.{src}", 'a') as csrc, open(f"{prefix}.rand{ntarget}.{tgt}", 'a') as ctgt:
        for line1 in buffer_processed_src:
            csrc.write(line1)
        for line2 in buffer_processed_tgt:
            ctgt.write(line2)
    buffer_processed_src = []
    buffer_processed_tgt = []
