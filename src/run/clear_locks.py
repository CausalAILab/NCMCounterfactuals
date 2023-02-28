import argparse
import os


def del_locks(dir):
    deleted = 0
    for item in os.listdir(dir):
        next_dir = "{}/{}".format(dir, item)
        if item == "lock":
            if args.verbose:
                print("Removing lock from {}".format(dir))
            os.remove(next_dir)
            deleted += 1
        elif os.path.isdir(next_dir):
            deleted += del_locks(next_dir)
    return deleted


parser = argparse.ArgumentParser(description="Clear all locks from specified directory.")
parser.add_argument('dir', help="directory to scan")
parser.add_argument('--verbose', action="store_true", help="print more information")

args = parser.parse_args()
d = "./{}".format(args.dir)
print("{} locks deleted.".format(del_locks(d)))
