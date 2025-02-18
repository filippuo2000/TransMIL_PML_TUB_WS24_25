from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd


def heatmap(
    slide_id: str = None,
    patch_id_list=list[int],
    base_dir="/mnt/",
    save_dir="./",
):
    if slide_id:
        slide_id = slide_id
        wsi_path = Path(base_dir, "slides", f"{slide_id}.tif")
        csv_path = Path(
            base_dir, "patches", "20x", slide_id, "metadata", "df.csv"
        )
        label_path = Path(base_dir, "splits/camelyon16_tumor_85_15_orig_0.csv")

    df_labels = pd.read_csv(label_path)
    wsi_label = (
        df_labels['label'].loc[df_labels['case_id'] == slide_id].tolist()[0]
    )

    if wsi_label == 0:
        label = "No Tumor"
    elif wsi_label == 1:
        label = "Tumor"

    # Load the WSI
    slide = openslide.OpenSlide(wsi_path)

    # Select the lowest resolution level for visualization
    level = slide.level_count - 4
    wsi_dimensions = slide.level_dimensions[level]
    wsi_thumbnail = slide.read_region(
        location=(0, 0), level=level, size=wsi_dimensions
    )
    wsi_thumbnail = wsi_thumbnail.convert("RGB")
    arr = np.array(wsi_thumbnail)
    black_arr = np.ones((arr.shape))
    print(f"thumbnail shape is {arr.shape}")

    # Downsampling factor for the selected level
    downsample_factor = slide.level_downsamples[level]

    # Load patch coordinates from the CSV
    df = pd.read_csv(csv_path)
    selected_patches = df

    # Plot the rotated WSI
    plt.figure(figsize=(12, 12))
    # plt.imshow(np.array(wsi_thumbnail))
    plt.imshow(black_arr)
    plt.axis("off")

    # norm = Normalize(vmin=0, vmax=1)

    for i, row in selected_patches.iterrows():
        # Parse position_abs (convert string "(x, y)" to tuple)
        position_abs = eval(
            row["position_abs"]
        )  # Example: "(512, 10240)" -> (512, 10240)
        x_abs, y_abs = position_abs
        x_scaled = x_abs / downsample_factor
        y_scaled = y_abs / downsample_factor

        # Get patch size
        patch_size_abs = row["patch_size_abs"]
        patch_size_scaled = patch_size_abs / downsample_factor

        importance = patch_id_list[i]
        curr_color = plt.cm.Reds(importance)
        # Draw the rectangle
        rect = patches.Rectangle(
            (x_scaled, y_scaled),
            patch_size_scaled,
            patch_size_scaled,
            color=curr_color,
            linewidth=0,
            facecolor=curr_color,
            alpha=1.0,
        )

        plt.gca().add_patch(rect)

    plt.title(
        f"WSI with Selected Patches Highlighted\
        - slide {slide_id}, label: {label}"
    )
    save_path = Path(save_dir, f"heatmap_{slide_id}.png")
    plt.savefig(save_path)
    plt.show()
