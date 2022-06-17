import os
import sys
import argparse

sys.path.append(os.path.abspath("src"))

from detection import analyse


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "path",
        help = "Path to image or directory of images.",
        type = str
    )
    
    return parser.parse_args()

def main() -> None:
    args = process_args()
    
    if args.path is None:
        raise FileNotFoundError("Path provided is incorrect.")
    
    results_dir = os.path.join(os.path.dirname(args.path), "results")
    
    analyse(args.path, results_dir)
    
if __name__ == "__main__":
    main()

        
        