import polars as pl


def read_base() -> pl.LazyFrame:
    df1 = pl.scan_csv("./data/SO_QuotationBas.csv")
    df2 = pl.scan_csv("./data/SO_QuotationBas1.csv")

    sch1 = df1.collect_schema()
    df2 = df2.with_columns([pl.col(col).cast(sch1[col]) for col in sch1])
    df = (
        pl.concat([df1, df2])
        .filter(pl.col("DataType") == 1)
        .select(
            "ContractCode",
            "TradingDate",
            "UnderlyingSecuritySymbol",
            "ClosePrice",
            "Position",
            "Volume",
            "Amount",
            "SettlePrice",
        )
    )

    return df


def read_der() -> pl.LazyFrame:
    df1 = pl.scan_csv("./data/SO_QuotationDer.csv")
    df2 = pl.scan_csv("./data/SO_QuotationDer1.csv")

    sch1 = df1.collect_schema()
    df2 = df2.with_columns([pl.col(col).cast(sch1[col]) for col in sch1])
    df = (
        pl.concat([df1, df2])
        .filter(pl.col("DataType") == 1)
        .select(
            "ContractCode",
            "TradingDate",
            "MainSign",
            "ContinueSign",
            "CallOrPut",
        )
    )

    return df


def read_para() -> pl.LazyFrame:
    df1 = pl.scan_csv("./data/SO_PricingParameter.csv")
    df2 = pl.scan_csv("./data/SO_PricingParameter1.csv")

    sch1 = df1.collect_schema()
    df2 = df2.with_columns([pl.col(col).cast(sch1[col]) for col in sch1])
    df = (
        pl.concat([df1, df2])
        .filter(pl.col("DataType") == 1)
        .select(
            "ContractCode",
            "TradingDate",
            "StrikePrice",
            "ExerciseDate",
            "HistoricalVolatility",
            "ImpliedVolatility",
            "Delta",
            "Gamma",
            "Theta",
            "Vega",
            "Rho",
        )
    )

    return df


def read_data():
    base_df = read_base()
    der_df = read_der()
    para_df = read_para()

    df = (
        base_df.join(der_df, on=["ContractCode", "TradingDate"], how="inner")
        .join(para_df, on=["ContractCode", "TradingDate"], how="inner")
        .filter(pl.col("UnderlyingSecuritySymbol") == 510050)
        .sort(["TradingDate", "StrikePrice", "ExerciseDate", "CallOrPut"])
        .with_columns(
            pl.all()
            .exclude("TradingDate", "ContractCode", "CallOrPut", "ExerciseDate")
            .cast(pl.Float64, strict=False),
            pl.col("TradingDate", "ExerciseDate").cast(pl.Date, strict=False),
            pl.col("ContractCode").str.replace_all(r"[AB]", "M"),
        )
        .collect()
    )

    return df


def read_etf():
    df = (
        pl.scan_csv("./data/FUND_MKT_Quotation.csv")
        .with_columns(pl.col("TradingDate").cast(pl.Date, strict=False))
        .with_columns(pl.col("ChangeRatio").cast(pl.Float64, strict=False))
        .select("TradingDate", "ClosePrice", "ChangeRatio", "Volume", "Amount")
        .sort("TradingDate")
        .collect()
    )

    return df


def read_combined() -> pl.DataFrame:
    data = read_data()
    etf = read_etf()

    data = (
        data.join(etf, on="TradingDate", how="left", suffix="_ETF")
        .with_columns(
            (pl.col("StrikePrice") - pl.col("ClosePrice_ETF")).abs().alias("Diff")
        )
        .with_columns(
            pl.col("Diff")
            .abs()
            .rank()
            .over("TradingDate", "CallOrPut", "ExerciseDate")
            .alias("DiffRank")
        )
        .with_columns(
            pl.when(pl.col("DiffRank") <= 1).then(1).otherwise(0).alias("ATMFlag")
        )
    )

    return data


if __name__ == "__main__":
    read_data()
