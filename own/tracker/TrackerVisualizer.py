import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

def TrackerVisualizer(FrameList):
    frames = FrameList.frames
    tot_frames = len(frames)
    plt.figure()
    i = 0
    display_image = True

    colors = ['r', 'b', 'g', 'y', 'c']
    while display_image:
        plt.imshow(cv2.cvtColor(frames[i].rgb_image, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        for obj in frames[i].object_list:
            rect = patches.Rectangle((obj.pos_x, obj.pos_y), obj.box_width, obj.box_height, edgecolor=colors[obj.id%len(colors)],
                                     facecolor="none", linewidth=3)
            plt.text(obj.pos_x, obj.pos_y-10, str(obj.id), c = colors[obj.id%len(colors)])
            ax.add_patch(rect)
        plt.pause(0.0001)
        plt.clf()
        i += 2
        if i >= tot_frames:
            display_image = False