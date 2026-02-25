import sys
import os
import shutil

sys.path.append(os.path.abspath("./"))

from tools import collect_env

def main():
    # check torch install
    collect_env.main()

    columns, row = shutil.get_terminal_size(fallback=(80, 24))
    print("="*columns)
    print(r"""
    ____    __                    __                          ___ 
   / __/   / /  ____ _   _____   / /_   ____   _____  _____  |__ \
  / /_    / /  / __ `/  / ___/  / __ \ / __ \ / ___/ / ___/  __/ /
 / __/   / /  / /_/ /  (__  )  / / / // /_/ // /__  / /__   / __/ 
/_/     /_/   \__,_/  /____/  /_/ /_/ \____/ \___/  \___/  /____/ 
                                                                  
""")
    print("="*columns)

if __name__ == "__main__":
    main()
