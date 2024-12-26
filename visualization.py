from pathlib import Path
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd


def plot_stats(
    epochs,
    y_vals,
    plot_name,
    plot_type=["loss", "acc"],
    save=False,
    save_path="",
):
    if plot_type not in ['loss', 'acc']:
        raise ValueError(f"Type {plot_type} is not valid for plotting")
    x = list(range(1, epochs + 1))
    y = y_vals

    plt.plot(x, y)
    plt.title(plot_name)
    plt.xlabel("Epoch")
    plt.ylabel(plot_type)

    if save:
        save_path = Path(
            save_path, f"{plot_type}_{plot_name}_{epochs}_epochs.png"
        )
        plt.savefig(
            save_path, bbox_inches="tight", dpi=300
        )  # High resolution and tight layout
        print(f"Plot saved to {save_path}")
    plt.show()
    plt.clf()


def visualize_slide_with_patches(
    slide_id: str, outfile: Path = None, base_dir="/mnt/"
):
    # Prepare path names
    if slide_id:
        slide_id = slide_id
        # WSI filepath
        wsi_path = Path(base_dir, "slides", f"{slide_id}.tif")
        # WSI's patches datafile contain info about their position in the WSI
        csv_path = Path(
            base_dir, "patches", "20x", slide_id, "metadata", "df.csv"
        )
        # File containing WSI's label info
        label_path = Path(base_dir, "splits/camelyon16_tumor_85_15_orig_0.csv")
    else:
        raise ValueError("No slide_id has been given to the function")

    df_labels = pd.read_csv(label_path)
    wsi_label = (
        df_labels['label'].loc[df_labels['case_id'] == slide_id].tolist()[0]
    )

    # Assign string labels instead of int
    if wsi_label == 0:
        label = "No Tumor"
    elif wsi_label == 1:
        label = "Tumor"

    # Load the WSI
    slide = openslide.OpenSlide(wsi_path)

    # Adjusting resolution level for visualization
    level = slide.level_count - 3
    wsi_dimensions = slide.level_dimensions[level]
    wsi_thumbnail = slide.read_region(
        location=(0, 0), level=level, size=wsi_dimensions
    )
    wsi_thumbnail = wsi_thumbnail.convert("RGB")

    # Downsampling factor for the selected level
    downsample_factor = slide.level_downsamples[level]

    # Check if WSI is vertical (rotate if necessary)
    rotated = False
    if wsi_dimensions[1] > wsi_dimensions[0]:
        print("Rotating WSI to horizontal position.")
        wsi_thumbnail = wsi_thumbnail.rotate(-90, expand=True)
        rotated = True

    # Load patch coordinates from the CSV
    df = pd.read_csv(csv_path)

    # Randomly sample N patches and put their ids on the list
    N = 5
    selected_patches = df.sample(n=N).reset_index()
    selected_patches_ids = selected_patches['path'].tolist()

    # Plot the rotated WSI
    plt.figure(figsize=(12, 12))
    plt.imshow(np.array(wsi_thumbnail))
    plt.axis("off")

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

        if rotated:
            # Adjust coordinates for a 90° clockwise rotation
            x_scaled_new = wsi_dimensions[1] - y_scaled - patch_size_scaled
            y_scaled_new = x_scaled
            x_scaled, y_scaled = x_scaled_new, y_scaled_new

        # Draw the rectangle
        rect = patches.Rectangle(
            (x_scaled, y_scaled),
            patch_size_scaled,
            patch_size_scaled,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
        )
        plt.gca().add_patch(rect)
        plt.text(x_scaled, y_scaled - 5, f"{i+1}", color='black', fontsize=10)

    plt.title(
        f"Rotated WSI with Selected Patches Highlighted \
            - slide {slide_id}, label: {label}"
    )
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

    # Show single patches randomly chosen earlier
    show_patches(patch_ids=selected_patches_ids, rows=1, cols=5)


def show_patches(
    patch_ids: List[str], rows: int, cols: int, outfile: Optional[Path] = None
):
    if len(patch_ids != rows * cols):
        raise ValueError(
            f"{len(patch_ids)} patches cannot be ploted \
                in {rows} rows and {cols} columns grid"
        )
    imgs = patch_ids

    figure, axes = plt.subplots(rows, cols, figsize=(7, 7))
    axes = axes.flatten()
    # Add single patches to the plot
    for idx, ax in enumerate(axes):
        if idx < len(imgs):
            img = plt.imread(Path("/mnt/patches/20x/", imgs[idx]))
            ax.set_title(idx + 1, fontsize='8')
            ax.imshow(img)
            ax.axis('off')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
