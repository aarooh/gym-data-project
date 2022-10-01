import click

from pathlib import Path

import pandas as pd
from configs.config import get_config
import pytest
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

warnings.filterwarnings("ignore", category=UserWarning)


def load_dataset(config) -> pd.DataFrame:
    """
    Loads datasets from config file

    Args:
        config (Config): Pipeline configuration
    Returns:
        gym_df (pd.DataFrame): Gym data
        weather_df (pd.DataFrame): Weather data
    """
    gym_df_path = Path(f"./data/{config.gym_data_name}").absolute()
    gym_df = pd.read_csv(gym_df_path)

    weather_df_path = Path(f"./data/{config.weather_data_name}").absolute()
    weather_df = pd.read_csv(weather_df_path)
    return gym_df, weather_df


def aggregate_data(df) -> pd.DataFrame:
    """
    Aggregates data by hour

    All activity values under 2 are rounded up to 2 so value 2 means
    the equipment was occupied 0-20% of a 10 minutes period.
    After aggregation this will lead cumulative error after aggregation.

    Args:
        df (pd.DataFrame): Dataframe to aggregate
    Returns:
        df (pd.DataFrame): Aggregated dataframe
    """
    df["time"] = pd.to_datetime(df["time"])
    df = df.groupby(pd.Grouper(key="time", axis=0, freq="H")).sum()
    return df


def rename_columns(df, config):
    """
    Renames columns to more readable names

    Args:
        df (pd.DataFrame): Dataframe
        config (Config): Pipeline configuration
    Returns:
        df (pd.DataFrame): Dataframe
    """
    device_mapping = config.device_mapping
    df = df.rename(columns=device_mapping)
    return df


def print_head(df, n=10):
    """
    Prints head of dataframe

    Args:
        df (pd.DataFrame): Dataframe to print
        n (int): Number of rows to print
    Returns:
        df (pd.DataFrame): Dataframe
    """
    print(f"Present {n} first rows of the dataset.")
    print(df.head(n))
    return df


def run_tests(df):
    """
    Run pytest tests
    These tests are located in tests/test_data.py
    They can also be ran with pytest command

    Args:
        df (pd.DataFrame): Dataframe
    Returns:
        df (pd.DataFrame): Dataframe
    """
    pytest.main()
    pytest.main(["-x", "tests"])
    return df


def add_weekday(df):
    """
    Adds weekday as number to column

    Args:
        df (pd.DataFrame): Dataframe
    Returns:
        df (pd.DataFrame): Dataframe
    """
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "time"})
    df["weekday"] = pd.DatetimeIndex(df["time"]).dayofweek

    return df


def add_sum_of_minutes(df):
    """
    Sums minutes of each equipment to get total minutes of use

    Args:
        df (pd.DataFrame): Dataframe
    Returns:
        df (pd.DataFrame): Dataframe
    """
    cols_to_sum = df.columns[1:]
    df["total_minutes"] = df[cols_to_sum].sum(axis=1)
    return df


def add_hour_as_number(df):
    """
    Adds hour as number to column

    Args:
        df (pd.DataFrame): Dataframe
    Returns:
        df (pd.DataFrame): Dataframe
    """
    df["hour"] = pd.DatetimeIndex(df["time"]).hour
    return df


def most_popular_device(df, config):
    """
    Calculate most popular device by minutes used

    Args:
        df (pd.DataFrame): Dataframe
        config (Config): Pipeline configuration
    Returns:
        df (pd.DataFrame): Dataframe
    """
    list_of_devices = list(config.device_mapping.values())
    sum_df = df[list_of_devices].sum(axis=0)
    print(
        f"Most popular device is {sum_df.idxmax()} with {sum_df.max()} minutes of use in total."
    )
    return df


def draw_line_plot(df, x, y, title, path):
    """
    Draws plot

    Args:
        df (pd.DataFrame): Dataframe
        x (str): Column name for x axis
        y (str): Column name for y axis
        title (str): Title for plot
        path (Path): Path to save plot
    Return:
        df (pd.DataFrame): Dataframe
    """
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.savefig(f"{path}/{title}-lineplot.png")
    print(f"Saved plot {title} to plots folder.")
    return df


def plot_predictions(df, x, y, title, path):
    """Plot Actual and predicted values to see if they have linear relationship.

    Args:
        df (pd.DataFrame): Dataframe
        x (str): Column name for x axis
        y (str): Column name for y axis
        title (str): Title for plot
        path (Path): Path to save plot

    Return:
        df (pd.DataFrame): Dataframe
    """

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.lmplot(data=df, x=x, y=y)
    line_coords = np.arange(
        df["total_minutes"].min().min(), df["total_minutes"].max().max()
    )
    plt.plot(line_coords, line_coords, color="darkred", linestyle="--")
    plt.title(title)
    plt.savefig(f"{path}/{title}-scatterplot.png")
    print(f"Saved plot {title} to plots folder.")
    return df


def draw_scatter_plot(df, x, y, title, path):
    """
    Draws scatterplot

    Args:
        df (pd.DataFrame): Dataframe
        x (str): Column name for x axis
        y (str): Column name for y axis
        title (str): Title for plot
        path (Path): Path to save plot
    Return:
        df (pd.DataFrame): Dataframe
    """
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.regplot(
        data=df, x=x, y=y, ax=ax, line_kws={"color": "r", "alpha": 0.7, "lw": 5}
    )
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.savefig(f"{path}/{title}-scatterplot.png")
    print(f"Saved plot {title} to plots folder.")
    return df


def make_sum_df(df, list_of_cols, group_by_col):
    """
    Makes sum of columns

    Args:
        df (pd.DataFrame): Dataframe
        list_of_cols (list): List of columns to sum
        group_by_col (str): Column name to group by
    Returns:
        sum_df (pd.DataFrame): Aggregated dataframe by group_by_col
    """
    return df.groupby(group_by_col)[list_of_cols].agg("mean")


def most_popular_time_of_day(df, config):
    """
    Calculate most popular time of day by minutes used

    Args:
        df (pd.DataFrame): Dataframe
        config (Config): Pipeline configuration
    Returns:
        df (pd.DataFrame): Dataframe
    """
    list_of_cols = list(config.device_mapping.values())
    list_of_cols.append("total_minutes")
    graph_path = Path("./graphs").absolute()
    draw_line_plot(
        df=df,
        x="hour",
        y="total_minutes",
        title="Most_popular_time_of_day",
        path=graph_path,
    )
    draw_scatter_plot(
        df=df,
        x="hour",
        y="total_minutes",
        title="Most_popular_time_of_day",
        path=graph_path,
    )
    sum_df = make_sum_df(df, list_of_cols, "hour")
    print(f"Most popular time of day for {sum_df.idxmax()}")
    print(f"Least popular time of day for {sum_df.idxmin()}")
    print(f"Descriptive statistics for each hour {sum_df.describe()}")

    print("From the graph we can see that most popular time of day is 2PM - 4PM")
    print("Least popular time of day is 8PM - 4AM")

    corr = df["hour"].corr(df["total_minutes"])
    print(f"Correlation between hour and total minutes {round(corr, 4)}")

    print(
        "From the statistics and graphs we can declare that time of day has impact on the popularity of the gym"
    )
    return df


def most_popular_day(df, config):
    """
    Calculate most popular day by minutes used
    and is the gym more popular on weekdays or weekends

    Args:
        df (pd.DataFrame): Dataframe
        config (Config): Pipeline configuration
    Returns:
        df (pd.DataFrame): Dataframe
    """
    list_of_cols = list(config.device_mapping.values())
    list_of_cols.append("total_minutes")
    graph_path = Path("./graphs").absolute()
    draw_line_plot(
        df=df,
        x="weekday",
        y="total_minutes",
        title="Most_popular_time_of_week",
        path=graph_path,
    )
    draw_scatter_plot(
        df=df,
        x="weekday",
        y="total_minutes",
        title="Most_popular_time_of_week",
        path=graph_path,
    )
    sum_df = make_sum_df(df, list_of_cols, "weekday")
    print(f"Most popular day of the week for {sum_df.idxmax()}")
    print(f"Least popular day of the week for {sum_df.idxmin()}")
    print(f"Descriptive statistics for each hour {sum_df.describe()}")

    print(
        "From the graph we can see that most popular days are Mondays, Tuesdays and Wednesdays."
    )
    print("Least popular days are Fridays and Saturdays.")

    corr = df["weekday"].corr(df["total_minutes"])
    print(f"Correlation between weekday and total minutes {round(corr, 4)}")

    print(
        "From the statistics and graphs we can declare that the gym is more popular on weekdays than on weekends."
    )
    return df


def make_datime_column(df):
    """
    Makes datetime column from Year, Month, Day and Hour columns

    Args:
        df (pd.DataFrame): Dataframe
    Returns:
        df (pd.DataFrame): Dataframe
    """
    df["time"] = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str)
        + "-"
        + df["Day"].astype(str)
        + " "
        + df["Hour"].astype(str),
        utc=True,
    )
    return df


def temperature_impact(df):
    """
    Calculate if temperature has impact on the popularity of the gym

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        df (pd.DataFrame): Dataframe
    """
    graph_path = Path("./graphs").absolute()
    draw_scatter_plot(
        df=df,
        x="Temperature (degC)",
        y="total_minutes",
        title="Temperature_impact",
        path=graph_path,
    )
    corr = df["Temperature (degC)"].corr(df["total_minutes"])
    print(f"Correlation between Temperature and total minutes {round(corr, 4)}")
    print(
        "From the scatterplot we can see that the temperature has impact on the popularity of the gym."
    )
    print(
        "From the correlation we can see that the temperature has a weak positive correlation with the popularity of the gym."
    )
    return df


def precipitation_impact(df):
    """
    Calculate if precipitation has impact on the popularity of the gym

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        df (pd.DataFrame): Dataframe
    """

    # Findings: -1 = there was really no precipation or snow depth, but filtering it out has no impact on the results
    graph_path = Path("./graphs").absolute()
    draw_scatter_plot(
        df=df,
        x="Precipitation (mm)",
        y="total_minutes",
        title="Precipitation_impact",
        path=graph_path,
    )
    corr = df["Precipitation (mm)"].corr(df["total_minutes"])
    print(f"Correlation between Precipitation and total minutes {round(corr, 4)}")
    print(
        "From the scatterplot we can see that the amount of precipitation has negative impact on the popularity of the gym."
    )
    print(
        "From the correlation we can see that the precipitation has a weak negative correlation with the popularity of the gym."
    )
    return df


def make_predictions(df, config):
    """Use LinearRegression model to predict the popularity of the gym

    Args:
        df (pd.DataFrame): Dataframe
        config (Config): Pipeline configuration

    Returns:
        df (pd.DataFrame): Dataframe with predictions
    """

    # Load Model
    model_path = Path(config.model_path).absolute()
    model = joblib.load(model_path)
    # Fill NA values with 0
    df = df.fillna(0)
    # Create features DataFrame
    features = df[
        [
            "weekday",
            "hour",
            "Precipitation (mm)",
            "Snow depth (cm)",
            "Temperature (degC)",
        ]
    ]

    df["predictions"] = features.apply(
        lambda x: model.predict(x.values.reshape(1, -1))[0], axis=1
    )
    return df


def calculate_residuals(df):
    """
    Calculate residuals

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        df (pd.DataFrame): Dataframe
    """
    df["residuals"] = df["total_minutes"] - df["predictions"]

    return df


def model_evaluation(df, config):
    """
    Evaluate model

    Args:
        df (pd.DataFrame): Dataframe
        config (Config): Pipeline configuration

    Returns:
        df (pd.DataFrame): Dataframe
    """
    # Load Model
    model_path = Path(config.model_path).absolute()
    model = joblib.load(model_path)
    print("Model evaluation")
    print(
        "Model variables: weekday, hour, Precipitation (mm), Snow depth (cm), Temperature (degC)"
    )
    print(
        f"Model coefficients describe the relationship between predictor variables and the result: {model.coef_}"
    )
    print(f"Model intercept: {model.intercept_}")
    print(
        f"Mean absolute error: {mean_absolute_error(df['total_minutes'], df['predictions'])}"
    )
    print(
        f"Mean squared error: {mean_squared_error(df['total_minutes'], df['predictions'])}"
    )
    print(
        f"Root mean squared error: {np.sqrt(mean_squared_error(df['total_minutes'], df['predictions']))}"
    )
    print("Optimally we would want to minimize the error metrics.")
    print("The ")
    print(f"Maximum error: {max_error(df['total_minutes'], df['predictions'])}")
    print("Maximum error is huge compared to scale of the data.")
    print(f"R2 score: {r2_score(df['total_minutes'], df['predictions'])}")
    print("R2 score is close to 0, so the model is not very good.")

    print(
        "From the metrics and scatterplot we can see that the model is performing poorly."
    )
    df.to_csv("./data/predictions.csv", index=False)
    return df


@click.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to config file",
)
def run_pipeline(
    config_path: Path,
):
    config = get_config(config_path)

    gym_df, weather_df = load_dataset(config)

    # Testing and preprocessing data
    gym_df = (
        gym_df.pipe(run_tests)
        .pipe(rename_columns, config)
        .pipe(aggregate_data)
        .pipe(print_head, 10)
    )

    # Add more features
    gym_df = gym_df.pipe(add_sum_of_minutes).pipe(add_weekday).pipe(add_hour_as_number)
    # Analyzing data
    gym_df = (
        gym_df.pipe(most_popular_device, config)
        .pipe(most_popular_time_of_day, config)
        .pipe(most_popular_day, config)
    )
    # Preprocess weather data
    weather_df = weather_df.pipe(make_datime_column)

    # Merge dataframes
    df = pd.merge(
        gym_df,
        weather_df[
            ["Precipitation (mm)", "Snow depth (cm)", "Temperature (degC)", "time"]
        ],
        on="time",
    )

    # Analyzing data
    df = df.pipe(temperature_impact).pipe(precipitation_impact)

    # Make Predictions
    df = (
        df.pipe(make_predictions, config)
        .pipe(calculate_residuals)
        .pipe(
            plot_predictions,
            "total_minutes",
            "predictions",
            "Actual_vs_predicted",
            Path("./graphs").absolute(),
        )
        .pipe(model_evaluation, config)
    )


if __name__ == "__main__":
    run_pipeline()
