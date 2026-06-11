import json
import os
import argparse
import random

def parse_args():

    parser = argparse.ArgumentParser(description="Generate train/test file lists for DQM runs.")
    parser.add_argument("--inputPath", type=str, default="/pnfs/icarus/persistent/users/micarrig/DQM/", help="Path to the DQM data directory.")
    parser.add_argument("--train_pct", type=float, default=0.8, help="Percentage of data to use for training (default: 0.8).")
    parser.add_argument("-o", "--outputString", type=str, default="filelist", help="Output string to prepend to the generated file lists (default: 'filelist').")
    parser.add_argument("--redirector", action="store_true", help="Use xrootd redirector for file paths.")
    return parser.parse_args()


def main():

    args = parse_args()

    with open("availableRuns.json", "r") as f:
        availableRuns = json.load(f)
    
    good = availableRuns["good"]
    bad = availableRuns["bad"]

    goodFiles = []
    badFiles = []

    for run in good:
        for filename in os.listdir(f"{args.inputPath}/CI_build_lar_ci_{run}/reco/"):
            if filename.endswith(".root") and filename.startswith('DQM'):
                goodFiles.append(f"{args.inputPath}/CI_build_lar_ci_{run}/reco/{filename}")

    for run in bad:
        for filename in os.listdir(f"{args.inputPath}/CI_build_lar_ci_{run}/reco/"):
            if filename.endswith(".root") and filename.startswith('DQM'):
                badFiles.append(f"{args.inputPath}/CI_build_lar_ci_{run}/reco/{filename}")

    # Shuffle the file lists
    random.shuffle(goodFiles)
    random.shuffle(badFiles)

    train_good = goodFiles[:int(len(goodFiles) * args.train_pct)]
    test_good = goodFiles[int(len(goodFiles) * args.train_pct):]

    #currently only using good files for training, but can easily add bad files if desired
    train_files = train_good
    test_files = test_good + badFiles

    random.shuffle(train_files)
    random.shuffle(test_files)

    print("Saving train files:", len(train_files))
    print("Saving test files:", len(test_files))

    if args.redirector:
        train_files = [f"root://fndcadoor.fnal.gov/{file.replace('/pnfs/', '')}" for file in train_files]
        test_files = [f"root://fndcadoor.fnal.gov/{file.replace('/pnfs/', '')}" for file in test_files]

    with open(f"{args.outputString}_train.txt", "w") as f:
        for file in train_files:
            f.write(file + "\n")

    with open(f"{args.outputString}_test.txt", "w") as f:
        for file in test_files:
            f.write(file + "\n")


if __name__ == "__main__":

    main()