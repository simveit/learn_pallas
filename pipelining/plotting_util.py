import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_heatmap(times, BLOCK_SIZES, filename='heatmap.png'):
    # Extract unique values for b0 and b1
    b0_values = sorted(set(b[0] for b in BLOCK_SIZES))
    b1_values = sorted(set(b[1] for b in BLOCK_SIZES))

    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(b0_values), len(b1_values)))

    # Fill the heatmap data
    for i, b0 in enumerate(b0_values):
        for j, b1 in enumerate(b1_values):
            if (b0, b1) in times:
                heatmap_data[i, j] = times[(b0, b1)]
            else:
                heatmap_data[i, j] = np.nan  # Use NaN for missing combinations

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(heatmap_data, cmap='viridis')

    # Set up the axes
    ax.set_xticks(np.arange(len(b1_values)))
    ax.set_yticks(np.arange(len(b0_values)))
    ax.set_xticklabels(b1_values)
    ax.set_yticklabels(b0_values)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Time (seconds)", rotation=-90, va="bottom")

    # Add labels and title
    ax.set_xlabel("Block Size (b1)")
    ax.set_ylabel("Block Size (b0)")
    ax.set_title("Execution Time for Different Block Sizes")

    # Loop over data dimensions and create text annotations
    for i in range(len(b0_values)):
        for j in range(len(b1_values)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.4f}",
                               ha="center", va="center", color="w")

    fig.tight_layout()
    
    # Save the figure as an image file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

    print(f"Heatmap saved as {filename}")