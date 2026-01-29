import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Constants & Configuration ---
DATA_FILE = 'data.csv'  # Path to your CSV file
DATASET_NAME = 'Apples Dataset'  # Name of dataset to appear in the chart title
TRAINING_IMAGES_COUNT = 978  # Number of images to appear in the chart title
FIGURE_SIZE = (16, 8)
X_LABEL_ROTATION = 45
SAVE_PLOTS = False  # Set to True to save images instead of showing them


def load_and_clean_data(file_path):
    """
    Loads the dataset and standardizes the augmentation column.
    """
    try:
        df = pd.read_csv(file_path)

        # Specific column cleanup
        if 'augmentation' in df.columns:
            df['augmentation'] = df['augmentation'].str.strip().str.lower()

        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None


def draw_vertical_separators(ax, x_labels):
    """
    Draws dashed vertical lines between categories on the x-axis.
    """
    # Draw lines between each category (indices 0 to N-1)
    for i in range(len(x_labels) - 1):
        ax.axvline(i + 0.5, linestyle='--', color='gray', alpha=0.5)


def plot_metric_distribution(df, metric):
    """
    Generates and displays a boxplot for a specific metric.
    """
    # Filter data for the specific metric
    df_metric = df[df['metric'] == metric].copy()

    if df_metric.empty:
        print(f"No data found for metric: {metric}")
        return

    plt.figure(figsize=FIGURE_SIZE)

    # Create the boxplot
    ax = sns.boxplot(
        data=df_metric,
        x='augmentation',
        y='score',
        hue='model'
    )

    # Handle x-axis categories for vertical lines
    x_labels = sorted(df_metric['augmentation'].unique())
    draw_vertical_separators(ax, x_labels)

    # Titles and Labels
    # Uses the DATASET_NAME and TRAINING_IMAGES_COUNT constants
    plt.title(
        f'Score Distribution by Augmentation and Model ({metric})\n'
        f'{DATASET_NAME} ({TRAINING_IMAGES_COUNT} training images on Augmented Datasets)'
    )
    plt.xlabel('Augmentation Type')
    plt.ylabel('Score')
    plt.xticks(rotation=X_LABEL_ROTATION)
    plt.legend(title='Model')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if SAVE_PLOTS:
        filename = f"boxplot_{metric}_{DATASET_NAME.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
    else:
        plt.show()

    # Close figure to free memory
    plt.close()


def main():
    """
    Main execution function.
    """
    print(f"Processing data for: {DATASET_NAME}...")
    df = load_and_clean_data(DATA_FILE)

    if df is not None:
        # Get list of unique metrics to plot
        if 'metric' in df.columns:
            metrics = df['metric'].unique()
            print(f"Found metrics: {metrics}")

            for metric in metrics:
                print(f"Plotting metric: {metric}")
                plot_metric_distribution(df, metric)
        else:
            print("Error: 'metric' column missing from dataset.")


if __name__ == "__main__":
    main()
