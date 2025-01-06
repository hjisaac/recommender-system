from collections import Counter
from datetime import datetime
from functools import partial
from matplotlib import pyplot as plt
from src.settings import settings
from src.utils import convert_flat_dict_to_string


# TODO: Rethink the implementation of all this. And rely on the new utils first
#   For example we should try to use the new function `convert_flat_dict_to_string`...
# TODO: Improve the implementation of this functions by making them more generic


def generate_kwargs_based_name(prefix="", extension="", **kwargs):
    """
    Generate filename based on the provided kwargs. Useful to create
    checkpoints filename and figure names based on the config of the
    environment.

    Parameters:
        prefix (str): Optional prefix to add at the beginning of the filename.
        extension (str): Optional file extension to add at the end of the filename.
        **kwargs: Key-value pairs to be included in the filename.

    Returns:
        str: Generated filename with timestamp and provided keyword arguments.

    Examples:
        >>> generate_kwargs_based_name(prefix="checkpoint", extension="pt", lambda_=0.1, gamma=0.5, tau=0.2)
        'checkpoint_20231028-152433_lambda0.1_gamma0.5_tau0.2.pt'

        >>> generate_kwargs_based_name(prefix="plot", extension="svg", width=800, height=600)
        'plot_20231028-152433_width800_height600.svg'
    """

    # This is a special case that we want to support. And this is
    # due to the fact that `lambda` is a reserved word is python
    _lambda = kwargs.pop("lambda_", None) or kwargs.pop("_lambda", None)
    kwargs["lambda"] = _lambda

    prefix = prefix or ""
    extension = f".{extension}" if extension else ""

    # Create a readable timestamp
    readable_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Construct the filename with provided keyword arguments
    filename_parts = [readable_timestamp, f"lambda{_lambda}"]
    for key, value in kwargs.items():
        filename_parts.append(f"{key}{value}")

    if prefix:
        filename_parts.insert(0, prefix)
    if extension:
        filename_parts.append(extension)

    # Join all parts with underscores
    return "_".join(filename_parts)


generate_config_based_name = partial(
    generate_kwargs_based_name,
    _lambda=settings.als.HYPER_LAMBDA,
    gamma=settings.als.HYPER_GAMMA,
    tau=settings.als.HYPER_TAU,
    epochs=settings.als.HYPER_N_EPOCH,
    factors=settings.als.HYPER_N_FACTOR,
    # The dataset lines count
    input=settings.general.LINES_COUNT_TO_READ,
)


def get_plt_figure_path(figure_name, subdir=""):
    """
    Return a path to save the figure to
    """
    config_based_figure_name = generate_config_based_name(
        prefix=figure_name, extension=settings.figures.PLT_FIGURE_FORMAT
    )
    return (
        f"{settings.figures.PLT_FIGURE_FOLDER}/{subdir}/{config_based_figure_name}"
        if subdir
        else f"{settings.figures.PLT_FIGURE_FOLDER}/{config_based_figure_name}"
    )


def plot_data_item_distribution_as_hist(
    indexed_data,
    plot_title: str = "Ratings distribution",
    plot_xlabel: str = "Ratings",
    plot_ylabel: str = "Count",
):
    # TODO: Use this link to fix https://stackoverflow.com/questions/23246125/how-to-center-labels-in-histogram-plotx hist
    data_to_plot = []
    for user_id in indexed_data.id_to_user_bmap:
        for data in indexed_data.data_by_user_id__train[user_id]:
            data_to_plot.append(data[1])  # access the rating

    plt.figure(figsize=(10, 6))
    plt.hist(data_to_plot, bins=10, edgecolor="black")
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)

    plt.savefig(
        get_plt_figure_path("ratings_distribution"),
        format=settings.figures.PLT_FIGURE_FORMAT,
    )
    plt.show()


def plot_power_low_distribution(indexed_data):  # noqa
    user_degrees_occurrences = {
        user_id: len(indexed_data.data_by_user_id__train[user_id])
        for user_id in indexed_data.id_to_user_bmap
    }
    movie_degrees_occurrences = {
        item_id: len(indexed_data.data_by_item_id__train[item_id])
        for item_id in indexed_data.id_to_item_bmap
    }
    user_degrees_occurrences_counter = Counter(user_degrees_occurrences.values())
    movie_degrees_occurrences_counter = Counter(movie_degrees_occurrences.values())

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        user_degrees_occurrences_counter.keys(),
        user_degrees_occurrences_counter.values(),
        label="Users",
    )
    ax.scatter(
        movie_degrees_occurrences_counter.keys(),
        movie_degrees_occurrences_counter.values(),
        label="Movies",
    )
    ax.set_title("Degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.yscale("log")
    plt.xscale("log")

    plt.savefig(
        get_plt_figure_path("power_low"), format=settings.figures.PLT_FIGURE_FORMAT
    )

    plt.show()


def plot_als_train_test_rmse_evolution(als_model):  # noqa
    """
    Plots the error evolution over iterations for a given set of error values.

    Parameters:
    error_values: List or array of error values (e.g., RMSE, loss) for each iteration.
    label: Label for the plot (default is "Test MSE").
    """

    iterations = range(1, len(als_model._epochs_rmse_train) + 1)  # noqa

    # Plotting the error values
    plt.plot(
        iterations, als_model._epochs_rmse_train, label="Train RMSE", color="blue"
    )  # noqa
    plt.plot(
        iterations, als_model._epochs_rmse_test, label="Test RMSE", color="red"
    )  # noqa
    # Adding titles and labels
    plt.title("RMSE Evolution Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")

    # Show legend
    plt.legend()

    plt.savefig(get_plt_figure_path("rmse_test_train"))

    # Display the plot
    plt.show()


def plot_als_train_test_loss_evolution(als_model):  # noqa
    """
    Plots the loss evolution over iterations for the given ALS model.

    Parameters:
    als_model: Object containing 'epochs_loss_train' and 'epochs_loss_test'
               attributes that represent the loss values for each iteration.
    """
    iterations = range(1, len(als_model._epochs_loss_train) + 1)

    # Plotting the loss values
    plt.plot(iterations, als_model._epochs_loss_train, label="Train Loss", color="blue")
    plt.plot(iterations, als_model._epochs_loss_test, label="Test Loss", color="red")

    # Adding titles and labels
    plt.title("Loss Evolution Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    # Show legend
    plt.legend()
    plt.savefig(
        get_plt_figure_path("loss_test_train"),
        format=settings.figures.PLT_FIGURE_FORMAT,
    )

    # Display the plot
    plt.show()


def plot_error_evolution(
    error_values,
    label="Error",
    title="Error Evolution Over Iterations",
    ylabel="Error",
    color="blue",
):
    """
    Plots the error evolution over iterations for a given set of error values.

    Parameters:
    error_values: List or array of error values (e.g., RMSE, loss) for each iteration.
    label: Label for the plot (default is "Error").
    title: Title for the plot (default is "Error Evolution Over Iterations").
    ylabel: Y-axis label (default is "Error").
    """
    iterations = range(1, len(error_values) + 1)

    # Plotting the error values
    plt.plot(iterations, error_values, label=label, color=color)

    # Adding titles and labels
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)

    # Show legend
    plt.legend()
    plt.savefig(
        get_plt_figure_path(title.lower().replace(" ", "_")),
        format=settings.figures.PLT_FIGURE_FORMAT,
    )

    # Display the plot
    plt.show()
