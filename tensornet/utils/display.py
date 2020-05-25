import os
import matplotlib.pyplot as plt


def plot_metric(data, metric, legend_loc='lower right'):
    """Plot accuracy graph or loss graph.

    Args:
        data (list or dict): If only single plot then this is a list, else
            for multiple plots this is a dict with keys containing.
            the plot name and values being a list of points to plot
        metric (str): Metric name which is to be plotted. Can be either
            loss or accuracy.
        legend_loc (str, optional): Location of the legend box in the plot.
            No legend will be plotted if there is only a single plot.
            (default: 'lower right')
    """

    single_plot = True
    if type(data) == dict:
        single_plot = False
    
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    plt.title(f'{metric} Change')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot: # Set legend
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=legend_loc,
            shadow=True,
            prop={'size': 15}
        )

    # Save plot
    fig.savefig(f'{"_".join(metric.split()).lower()}_change.png')


def plot_predictions(data, classes, plot_title, plot_path):
    """Display data.

    Args:
        data (list): List of images, model predictions and ground truths.
            Images should be numpy arrays.
        classes (list or tuple): List of classes in the dataset.
        plot_title (str): Title for the plot.
        plot_path (str): Complete path for saving the plot.
    """

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(plot_title)

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        axs[row_count][idx % 5].imshow(result['image'])
    
    # Set spacing
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    # Save image
    fig.savefig(f'{plot_path}', bbox_inches='tight')


def save_and_show_result(classes, correct_pred=None, incorrect_pred=None, path=None):
    """Display network predictions.

    Args:
        classes (list or tuple): List of classes in the dataset.
        correct_pred (list, optional): Contains correct model predictions and labels.
            (default: None)
        incorrect_pred (list, optional): Contains incorrect model predictions and labels.
            (default: None)
        path (str, optional): Path where the results will be saved.
            (default: None)
    """

    # Create directories for saving predictions
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'predictions'
        )
    if not os.path.exists(path):
        os.makedirs(path)
    
    if not correct_pred is None:  # Plot correct predicitons
        plot_predictions(
            correct_pred, classes, 'Correct Predictions', f'{path}/correct_predictions.png'
        )

    if not incorrect_pred is None:  # Plot incorrect predicitons
        plot_predictions(
            incorrect_pred, classes, '\nIncorrect Predictions', f'{path}/incorrect_predictions.png'
        )
