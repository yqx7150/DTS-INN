import argparse

BATCH_SIZE = 1
FRAME = 18

# DATA_PATH = "./data/"


def get_arguments():
    parser = argparse.ArgumentParser(description="training codes")

    parser.add_argument("--frame", type=int, default=FRAME, help="Set the number of frames for dynamic PET data. ")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")
    parser.add_argument("--weight", type=float, default=1, help="Weight for forward loss. ")

    return parser
