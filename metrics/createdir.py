#!/usr/bin/python3
from zipfile import ZipFile
import os
import sys
import argparse
import shutil

def main():
    if len(sys.argv) != 4:
        print("python3 createDir.py <src_dir> <target_dir> <zip_file>")
        exit(0)
    
    my_parser = argparse.ArgumentParser(description="Create directory structure")
    my_parser.add_argument('spath', metavar='spath', type=str, help="Path of source dir to be copied")
    my_parser.add_argument('dpath', metavar='dpath', type=str, help="Path of destination dir")
    my_parser.add_argument('zpath', metavar='zpath', type=str, help="Path of zip file")
    
    args = my_parser.parse_args()
    
    spath = args.spath
    dpath = args.dpath
    zpath = args.zpath

    if not os.path.isdir(spath):
        print("Source directory path not exists")
        exit(0)
    
    if not os.path.isfile(zpath):
        print("Zip file not exists")
        exit(0)

    try:
        destination = shutil.copytree(spath, dpath)

        # Deleting info.txt if present
        info_path = os.path.join(destination, "info.txt")
        if os.path.isfile(info_path):
            os.remove(info_path)
        
        # Deleting the processed and recommendation files in the subdirectories
        recommendation_path = os.path.join(destination, "Recommendations", "RetrainFreq15Day")
        for sdir in os.listdir(recommendation_path):
            sdpath = os.path.join(recommendation_path, sdir)
            if os.path.isdir(sdpath):
                for file in os.scandir(sdpath):
                    if file.is_dir():
                        shutil.rmtree(file.path)
                    elif file.is_file():
                        os.remove(file.path)
        
        # Extracting the zip file into recommendation path
        with ZipFile(zpath) as zobj:
            zobj.extractall(path=recommendation_path)

        # rpath is the directory path of the Recommendation
        rpath = os.path.join(recommendation_path, "Recommendation")
        for file in os.scandir(rpath):
            if file.name.startswith("recommend_with_"):
                shutil.move(file.path, os.path.join(recommendation_path, "WithErrorModels", file.name))
            elif file.name.startswith("recommend_without_"):
                shutil.move(file.path, os.path.join(recommendation_path, "WithoutErrorModels", file.name))
            elif file.name == 'info.txt':
                shutil.move(file.path, info_path)
        
        # Removing the extracted recommendation directory path
        shutil.rmtree(rpath)
    except OSError as err:
        print("Error: ", err)

# if __name__ == "main":
#     main()

main()
