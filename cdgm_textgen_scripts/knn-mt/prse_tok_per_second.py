import argparse

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--hyp", help="hypothesis file")

    args = argp.parse_args()

    num_tokens = []
    seconds = []

    with open(args.hyp, "r") as f:
        line = f.readline()
        line = line.strip("\n")
        if line.startswith("W-"):
            line.split(" ")
            sec = float(line[1:2])
            seconds.append(sec)
        if line.startswith("H-"):
            line.split(" ")
            num_tokens.append(len(line[1:]))


    print("tok/sec", sum(num_tokens)/sum(seconds))




