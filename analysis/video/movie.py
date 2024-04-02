#! /usr/bin/python3
import cv2
import numpy as np
import skvideo.io
import skvideo.datasets
bbb = skvideo.datasets.bigbuckbunny()
import argparse
import json as json
import os, sys


def main():
    mod_name = str(sys.argv[1])
    vid_name = str(sys.argv[2])

    json_loc='/home/vturino/PhD/projects/exhumation/pyInput/'


    image_folder=f'/home/vturino/PhD/projects/exhumation/plots/single_models/{mod_name}/{vid_name}'
    out_folder=f'/home/vturino/PhD/projects/exhumation/plots/single_models/{mod_name}/videos/'
    if not os.path.exists(out_folder):
            os.mkdir(out_folder)
    
    fnumber = len(os.listdir(image_folder))


    out_video =  np.empty([fnumber, 1895,2697,3], dtype = np.uint8)
    out_video =  out_video.astype(np.uint8)

    for i in range(1, fnumber):
        img = cv2.imread(f"{image_folder}/{i}" + '.png')
        # print(f"{image_folder}/{i}" + '.png')
        out_video[i] = img

    fps = str('4')
    # Writes the the output image sequences in a video file
    skvideo.io.vwrite(f"{out_folder}{vid_name}.mp4", out_video, inputdict={'-r': fps, "-pix_fmt": "bgr24"}, outputdict={'-f': "mp4"})

if __name__ == "__main__":
    main()