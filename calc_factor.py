import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime


def calculate_atm(data: pl.DataFrame) -> pl.DataFrame:
    """
    计算每个交易日的平值期权隐含波动率（IV_ATM）
    """
    data_atm = (
        data.filter(pl.col("ATMFlag") == 1)
        # 保留每一个交易日的平值最大量的看涨和看跌期权
        .group_by("TradingDate", "CallOrPut")
        .agg(
            pl.col("ImpliedVolatility").sort_by("Amount").last().alias("IV"),
            pl.col("Vega").sort_by("Amount").last().alias("Vega"),
        )
        .group_by("TradingDate")
        .agg(
            (pl.col("IV") * pl.col("Vega")).sum().alias("IV"),
            pl.col("Vega").sum().alias("Vega"),
        )
        .with_columns((pl.col("IV") / pl.col("Vega")).alias("iv"))
        .select("TradingDate", "iv")
        .sort("TradingDate")
    )

    return data_atm


def calculate_term(data: pl.DataFrame) -> pl.DataFrame:
    """
    计算每个交易日的30天和90天平值期权隐含波动率（IV_30D, IV_90D）
    """
    data = (
        data.filter(pl.col("ATMFlag") == 1)
        .group_by("TradingDate", "CallOrPut")
        .agg(
            pl.col("ImpliedVolatility").sort_by("ExerciseDate").last().alias("IV_90"),
            pl.col("Vega").sort_by("ExerciseDate").last().alias("Vega_90"),
            pl.col("ImpliedVolatility").sort_by("ExerciseDate").first().alias("IV_30"),
            pl.col("Vega").sort_by("ExerciseDate").first().alias("Vega_30"),
        )
        .group_by("TradingDate")
        .agg(
            (pl.col("IV_90") * pl.col("Vega_90")).sum().alias("IV_90D"),
            pl.col("Vega_90").sum().alias("Vega_90"),
            (pl.col("IV_30") * pl.col("Vega_30")).sum().alias("IV_30D"),
            pl.col("Vega_30").sum().alias("Vega_30"),
        )
        .with_columns(
            (pl.col("IV_30D") / pl.col("Vega_30")).alias("IV_30D"),
            (pl.col("IV_90D") / pl.col("Vega_90")).alias("IV_90D"),
        )
        .with_columns(
            (pl.col("IV_30D") / pl.col("IV_90D")).alias("term"),
        )
        .select("TradingDate", "term")
        .sort("TradingDate")
    )

    return data


def calculate_skew(data: pl.DataFrame) -> pl.DataFrame:
    """
    计算每个交易日的最大交易量25delta看跌/看涨虚值期权IV之比
    """
    data = (
        data.group_by("TradingDate", "CallOrPut")
        .agg(
            pl.col("ImpliedVolatility")
            .sort_by((pl.col("Diff") / pl.col("StrikePrice") - 0.25).abs())
            .first()
            .alias("IV_25D_P"),
            pl.col("ImpliedVolatility")
            .sort_by((pl.col("Diff") / pl.col("StrikePrice") + 0.25).abs())
            .first()
            .alias("IV_25D_M"),
        )
        .group_by("TradingDate")
        .agg(
            pl.col("IV_25D_P").mean().alias("P"),
            pl.col("IV_25D_M").mean().alias("M"),
        )
        .with_columns((pl.col("M") / pl.col("P")).alias("skew"))
        .select("TradingDate", "skew")
        .sort("TradingDate")
    )

    return data


def calculate_pcr(data: pl.DataFrame) -> pl.DataFrame:
    data = (
        data.group_by("TradingDate")
        .agg(
            pl.col("Amount")
            .filter(pl.col("CallOrPut") == "P")
            .sum()
            .alias("PutAmount"),
            pl.col("Amount")
            .filter(pl.col("CallOrPut") == "C")
            .sum()
            .alias("CallAmount"),
        )
        .with_columns((pl.col("PutAmount") / pl.col("CallAmount")).alias("pcr"))
    )

    return data


def calculate_factor(data: pl.DataFrame) -> pl.DataFrame:
    data = data.filter(
        pl.col("ExerciseDate") - pl.col("TradingDate") >= pl.duration(days=28)
    ).sort(["TradingDate", "StrikePrice", "ExerciseDate", "CallOrPut"])

    data_atm = calculate_atm(data)
    data_term = calculate_term(data)
    data_skew = calculate_skew(data)
    data_pcr = calculate_pcr(data)

    data = (
        data_atm.join(data_term, on="TradingDate", how="inner")
        .join(data_skew, on="TradingDate", how="inner")
        .join(data_pcr, on="TradingDate", how="inner")
        .with_columns(
            pl.col("iv").rolling("TradingDate", period="1mo").rank().alias("rank"),
            pl.col("term").rolling("TradingDate", period="1mo").rank().alias("term"),
            pl.col("skew").rolling("TradingDate", period="1mo").rank().alias("skew"),
            pl.col("pcr").rolling("TradingDate", period="1mo").rank().alias("pcr"),
            pl.col("TradingDate")
            .rolling("TradingDate", period="1mo")
            .count()
            .alias("cnt"),
        )
        .with_columns(
            (pl.col("rank") / pl.col("cnt")).alias("rank"),
            (pl.col("term") / pl.col("cnt")).alias("term"),
            (pl.col("skew") / pl.col("cnt")).alias("skew"),
            (pl.col("pcr") / pl.col("cnt")).alias("pcr"),
        )
        .select("TradingDate", "iv", "rank", "term", "skew", "pcr")
        .filter(
            pl.col("TradingDate") >= pl.col("TradingDate").min() + pl.duration(days=365)
        )
        .interpolate()
        .drop_nulls()
        .sort("TradingDate")
    )

    draw_factor(data)

    return data


def draw_factor(data: pl.DataFrame):
    data = data.group_by_dynamic("TradingDate", every="7d", closed="left").agg(
        pl.col("iv").mean().alias("iv"),
        pl.col("rank").mean().alias("rank"),
        pl.col("term").mean().alias("term"),
        pl.col("skew").mean().alias("skew"),
        pl.col("pcr").mean().alias("pcr"),
    )

    plt.figure(figsize=(12, 8))

    for idx, col in enumerate(["rank", "term", "skew", "pcr"]):
        ax1 = plt.subplot(2, 2, idx + 1)

        # Plot the main factor on left y-axis
        color = "tab:blue"
        ax1.set_ylabel(col, color=color)
        ax1.plot(data["TradingDate"], data[col], label=col, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        # Create right y-axis for IV_ATM
        ax2 = ax1.twinx()
        color = "tab:orange"
        ax2.set_ylabel("iv", color=color)

        # Align means of both axes
        mean_col = data[col].mean()
        mean_iv = data["iv"].mean()
        col_range = data[col].max() - data[col].min()
        iv_range = data["iv"].max() - data["iv"].min()

        ax2.plot(
            data["TradingDate"],
            data["iv"],
            label="iv",
            color=color,
            linestyle="--",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Set limits to align means
        ax1.set_ylim(mean_col - col_range, mean_col + col_range)
        ax2.set_ylim(mean_iv - iv_range, mean_iv + iv_range)

        plt.title(f"{col} vs IV_ATM")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig("./output/option_factors.png")


def divide_data(
    data: pl.DataFrame, factors: pl.DataFrame, split_date: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dt = datetime.strptime(split_date, "%Y-%m-%d")

    train_data = data.filter(pl.col("TradingDate") < dt)
    test_data = data.filter(pl.col("TradingDate") >= dt)
    train_factor = factors.filter(pl.col("TradingDate") < dt)
    test_factor = factors.filter(pl.col("TradingDate") >= dt)

    return train_data, test_data, train_factor, test_factor
