import sys
import os
import shutil

sys.path.append(os.path.abspath("./"))

from tools import collect_torch_env

def main():
    # check torch install
    collect_torch_env.main()

    # check mm install
    pass

    columns, row = shutil.get_terminal_size(fallback=(80, 24))
    print("="*columns)
    print("Hello flashocc2!")

if __name__ == "__main__":
    main()
