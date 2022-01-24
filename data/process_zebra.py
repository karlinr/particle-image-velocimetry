import plistlib
import os
import numpy as np
import tifffile as tf

with open("zebrafish/unprocessed/params.plist", 'rb') as fp:
    plist = plistlib.load(fp)["piv"]["pivGroupings"][6]

for g, grouping in enumerate(plist):
    video_array = np.zeros((len(grouping) * 2, tf.imread(f"./zebrafish/unprocessed/030298.tif").shape[1],
                            tf.imread(f"./zebrafish/unprocessed/030298.tif").shape[2]))
    f = 0
    for video in os.listdir("./zebrafish/unprocessed/"):
        if video.endswith(".tif"):
            current_video = tf.imread(f"./zebrafish/unprocessed/{video}")
            for frame in grouping:
                lower_index = int(os.path.splitext(video)[0])
                upper_index = int(os.path.splitext(video)[0]) + 1000
                if lower_index <= frame < upper_index:
                    print(f)
                    print(g)
                    print(video)
                    print(frame - lower_index)
                    video_array[f] = current_video[frame - lower_index]
                    video_array[f + 1] = current_video[frame - lower_index + 1]
                    f += 2
    tf.imwrite(f"./zebrafish/processed/{g}.tif", video_array.astype(np.ushort))

print("Finished")