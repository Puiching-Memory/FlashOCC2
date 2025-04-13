import os
import sys
import shutil

def delete_pycache(root_dir):
    """删除指定目录及其子目录下的所有__pycache__文件夹"""
    try:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if '__pycache__' in dirnames:
                pycache_dir = os.path.join(dirpath, '__pycache__')
                # 删除__pycache__目录及其内容
                shutil.rmtree(pycache_dir, ignore_errors=True)
                print(f"Deleted: {pycache_dir}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python delete_pycache.py /path/to/directory")
        sys.exit(1)

    root_dir = sys.argv[1]
    
    # 检查路径是否存在且是目录
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} does not exist.")
        sys.exit(1)
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory.")
        sys.exit(1)

    print(f"Starting to delete __pycache__ folders in {root_dir}...")
    delete_pycache(root_dir)
    print("Done.")

if __name__ == "__main__":
    main()