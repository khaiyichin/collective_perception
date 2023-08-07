import matplotlib.pyplot as plt
import numpy as np
import argparse

# labels = ["U(0.525, 0.975)", "0.525", "0.575", "0.625", "0.675", "0.725", "0.775", "0.825", "0.875", "0.925", "0.975"]
labels = ["U(0.525, 0.975)", "0.525", "0.675", "0.825", "0.975"]

# color_bar_img = [[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
color_bar_img = [[0, 1, 7, 13, 19]]
fig, ax = plt.subplots()

def draw_horizontal():
    ax.imshow(color_bar_img, aspect=0.35, cmap="nipy_spectral", vmin=0, vmax=19)

    # Modify tick labels
    ax.set_ylabel("Sensor Accuracies", fontsize=14, rotation=0, ha="right")
    ylabel_pos = ax.yaxis.get_label().get_position()
    # ax.yaxis.set_label_coords(ylabel_pos[0]-0.05, ylabel_pos[1] - 0.75)
    ax.yaxis.set_label_coords(ylabel_pos[0]-0.05, ylabel_pos[1] - 0.4)
    ax.set_yticks([])
    ax.set_xticks(list(range(len(labels))), labels, fontsize=10, rotation=15, ha="right")

    fig.savefig("h_colorbar.png", bbox_inches="tight", dpi=300)

def draw_vertical():
    color_bar_img_T = [[i] for i in color_bar_img[0]]
    # ax.imshow(color_bar_img_T, aspect=2.5, cmap="nipy_spectral", vmin=0, vmax=19)
    ax.imshow(color_bar_img_T, aspect=4, cmap="nipy_spectral", vmin=0, vmax=19)

    # Modify tick labels
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel("Sensor\nAccuracies", fontsize=14, rotation=0, ha="left")
    xlabel_pos = ax.xaxis.get_label().get_position()
    ax.xaxis.set_label_coords(xlabel_pos[0]-0.5, xlabel_pos[1]-0.05)
    ax.set_xticks([])
    ax.set_yticks(list(range(len(labels))), labels, fontsize=10, rotation=30, ha="left", va="baseline")

    fig.savefig("v_colorbar.png", bbox_inches="tight", dpi=300)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-col", action="store_true")
    parser.add_argument("-row", action="store_true")

    args = parser.parse_args()

    if args.col: draw_vertical()
    else: draw_horizontal()

if __name__ == "__main__":
    main()
