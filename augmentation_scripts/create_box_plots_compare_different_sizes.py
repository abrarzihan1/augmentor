import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Constants & Configuration ---
DATA_FILE = "combined.csv"
DATASET_NAME = "Laboro Tomato Dataset"
NONE_AUGMENTATION_COUNT = 823  # Value to assign to 'none' augmentation image_count
FIGURE_SIZE = (12, 7)
X_LABEL_ROTATION = 15
COLOR_PALETTE_NAME = "Set2"
SAVE_PLOTS = False  # Set to True to save images, False to show them


def load_data(file_path):
    """
    Loads the dataset and standardizes the augmentation column.
    """
    try:
        df = pd.read_csv(file_path)
        # Clean string formatting
        if 'augmentation' in df.columns:
            df['augmentation'] = df['augmentation'].str.strip().str.lower()
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def prepare_data_for_plotting(df):
    """
    Consolidates 'none' augmentation counts and prepares data for plotting.
    """
    df_plot = df.copy()

    # For the 'none' augmentation, force the image_count to a specific representative value
    mask_none = df_plot['augmentation'] == 'none'
    df_plot.loc[mask_none, 'image_count'] = NONE_AUGMENTATION_COUNT

    return df_plot


def create_consistent_palette(df):
    """
    Creates a dictionary mapping unique image_counts to specific colors.
    Ensures colors remain consistent across different plots.
    """
    unique_counts = sorted(df['image_count'].unique())
    colors = sns.color_palette(COLOR_PALETTE_NAME, n_colors=len(unique_counts))
    return dict(zip(unique_counts, colors))


def customize_legend(ax, hidden_counts):
    """
    Filters the legend to exclude specific image counts (e.g., the 'none' category count).
    """
    handles, labels = ax.get_legend_handles_labels()

    # Filter out handles/labels where the label (image count) is in hidden_counts
    # Note: Labels from matplotlib are strings, so we cast to int for comparison
    filtered = [
        (h, l) for h, l in zip(handles, labels)
        if int(float(l)) not in hidden_counts
    ]

    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles, labels, title='Image Count')
    else:
        ax.legend().remove()


def plot_model_metric(df, model, metric, palette):
    """
    Generates a boxplot for a specific model and metric.
    """
    # Filter data
    plot_df = df[(df['model'] == model) & (df['metric'] == metric)]

    if plot_df.empty:
        return

    print(f"Generating plot for {model.upper()} with {metric}...")

    plt.figure(figsize=FIGURE_SIZE)

    ax = sns.boxplot(
        data=plot_df,
        x='augmentation',
        y='score',
        hue='image_count',
        palette=palette
    )

    # Draw vertical lines between categories
    x_labels = sorted(plot_df['augmentation'].unique())
    for i in range(len(x_labels) - 1):
        ax.axvline(i + 0.5, linestyle='--', color='gray', alpha=0.5)

    # Titles and Labels
    plt.title(f'Score Distribution for {model.upper()} ({metric}) on {DATASET_NAME}')
    plt.xlabel('Augmentation Type')
    plt.ylabel('Score')
    plt.xticks(rotation=X_LABEL_ROTATION)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Customize Legend: Hide the count used for 'none'
    customize_legend(ax, hidden_counts=[NONE_AUGMENTATION_COUNT])

    plt.tight_layout()

    if SAVE_PLOTS:
        filename = f"boxplot_{model}_{metric}_{DATASET_NAME.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
    else:
        plt.show()

    plt.close()


def main():
    print("--- Starting Processing ---")

    # 1. Load Data
    df = load_data(DATA_FILE)
    if df is None:
        return

    # 2. Prepare Data (Handle 'none' augmentation logic)
    df_plot = prepare_data_for_plotting(df)

    # 3. Create Palette
    palette = create_consistent_palette(df_plot)
    print(f"Color palette created for counts: {list(palette.keys())}")

    # 4. Generate Plots
    models = df_plot['model'].unique()
    metrics = df_plot['metric'].unique()

    for model in models:
        for metric in metrics:
            plot_model_metric(df_plot, model, metric, palette)

    print("--- Processing Complete ---")


if __name__ == "__main__":
    main()
