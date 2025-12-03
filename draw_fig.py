import matplotlib.pyplot as plt
import polars as pl
from datetime import datetime


def z_score(series: pl.Series) -> pl.Series:
    return (series - series.mean()) / series.std()


def draw_comparison(data: pl.DataFrame):
    data = (
        data.lazy()
        .filter(pl.col("MainSign") == 1)
        .select("TradingDate", "ImpliedVolatility", "ChangeRatio")
        .group_by_dynamic("TradingDate", every="14d", closed="left")
        .agg(
            pl.col("ImpliedVolatility").mean().alias("ImpliedVolatility"),
            pl.col("ChangeRatio").mean().alias("ChangeRatio"),
        )
        # z-score normalization
        .with_columns(
            z_score(pl.col("ImpliedVolatility")).alias("ImpliedVolatility"),
            z_score(pl.col("ChangeRatio")).alias("ChangeRatio"),
        )
        .collect()
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        data["TradingDate"],
        data["ImpliedVolatility"],
        label="Implied Volatility (z-score)",
        color="blue",
    )
    plt.plot(
        data["TradingDate"],
        data["ChangeRatio"],
        label="ETF Change Ratio (z-score)",
        color="orange",
    )
    plt.xlabel("Trading Date")
    plt.ylabel("Z-Score")
    plt.title("Comparison of Implied Volatility and ETF Change Ratio")
    plt.legend()
    plt.grid()
    plt.savefig("./output/implied_volatility_vs_etf_change_ratio.png")


def draw_mean_and_bollinger(data: pl.DataFrame):
    data = (
        data.lazy()
        .filter(pl.col("MainSign") == 1)
        .interpolate()
        .group_by_dynamic("TradingDate", every="5d", closed="left")
        .agg(
            pl.col("ImpliedVolatility").mean().alias("ImpliedVolatility"),
            pl.col("HistoricalVolatility").mean().alias("HistoricalVolatility"),
        )
        .with_columns(
            pl.col("ImpliedVolatility").rolling_mean(window_size=20).alias("MeanIV"),
            pl.col("ImpliedVolatility").rolling_std(window_size=20).alias("StdIV"),
        )
        .with_columns(
            (pl.col("MeanIV") + 2 * pl.col("StdIV")).alias("UpperBand"),
            (pl.col("MeanIV") - 2 * pl.col("StdIV")).alias("LowerBand"),
        )
        .select(
            "TradingDate",
            "ImpliedVolatility",
            "MeanIV",
            "UpperBand",
            "LowerBand",
            "HistoricalVolatility",
        )
        .collect()
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        data["TradingDate"],
        data["ImpliedVolatility"],
        label="Implied Volatility",
        color="blue",
    )
    plt.plot(
        data["TradingDate"],
        data["MeanIV"],
        label="100-Day Mean",
        color="orange",
    )
    plt.plot(
        data["TradingDate"],
        data["UpperBand"],
        label="Upper Bollinger Band",
        color="green",
        linestyle="--",
    )
    plt.plot(
        data["TradingDate"],
        data["LowerBand"],
        label="Lower Bollinger Band",
        color="red",
        linestyle="--",
    )
    plt.plot(
        data["TradingDate"],
        data["HistoricalVolatility"],
        label="Historical Volatility",
        color="purple",
    )
    plt.xlabel("Trading Date")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility with Bollinger Bands")
    plt.legend()
    plt.grid()
    plt.savefig("./output/implied_volatility_bollinger_bands.png")


def draw_violatility(data: pl.DataFrame, date: str):
    data = (
        data.filter(pl.col("TradingDate") == datetime.strptime(date, "%Y-%m-%d"))
        .filter(pl.col("ExerciseDate") > pl.col("TradingDate") + pl.duration(days=28))
        .group_by("StrikePrice", "ExerciseDate")
        .agg(
            (pl.col("ImpliedVolatility") * pl.col("Vega").abs()).sum().alias("T"),
            pl.col("Vega").abs().sum().alias("C"),
        )
        .with_columns((pl.col("T") / pl.col("C")).alias("IV"))
        .with_columns(pl.col("StrikePrice").cast(pl.Float64))
        .select("StrikePrice", "ExerciseDate", "IV")
        .sort("StrikePrice", "ExerciseDate")
    )

    plt.figure(figsize=(8, 6))
    for ex_date in data["ExerciseDate"].unique():
        ex_data = data.filter(pl.col("ExerciseDate") == ex_date)
        plt.plot(
            ex_data["StrikePrice"],
            ex_data["IV"],
            label=f"Exercise Date: {ex_date}",
        )
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title(f"Implied Volatility on {date}")
    plt.legend()
    plt.grid()
    plt.savefig(f"./output/implied_volatility_{date}.png")


def draw_fig(data: pl.DataFrame):

    draw_comparison(data)
    draw_mean_and_bollinger(data)
    draw_violatility(data, "2024-03-25")
    draw_violatility(data, "2024-09-30")
    draw_violatility(data, "2024-10-08")
