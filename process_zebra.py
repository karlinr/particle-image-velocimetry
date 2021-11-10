import plistlib
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import time

with open("data/zebrafish/params.plist", 'rb') as fp:
    plist = plistlib.load(fp)["piv"]["pivGroupings"][6]

for g, grouping in enumerate(plist):
    video_array = np.zeros((len(grouping) * 2, tf.imread(f"data/zebrafish/030298.tif").shape[1],
                            tf.imread(f"data/zebrafish/030298.tif").shape[2]))
    f = 0
    for video in os.listdir("data/zebrafish/"):
        if video.endswith(".tif"):
            current_video = tf.imread(f"data/zebrafish/{video}")
            for frame in grouping:
                lower_index = int(os.path.splitext(video)[0])
                upper_index = int(os.path.splitext(video)[0]) + 1000
                if lower_index <= frame < upper_index:
                    video_array[f] = current_video[frame - lower_index]
                    video_array[f + 1] = current_video[frame - lower_index + 1]
                    f += 2
    tf.imwrite(f"data/processedzebra/{g}_testdata.tif", video_array.astype(np.int16))

print("Finished")