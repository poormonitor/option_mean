import polars as pl
import itertools
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class BackTrace:
    def __init__(
        self, data: pl.DataFrame, factors: pl.DataFrame, balance: int, para: dict
    ):
        self.factors = factors
        self.para = para
        self.balance = balance
        self.fee = 8
        self.position = {}
        self.value = []

        self.exercise = {
            row["ContractCode"]: row["ExerciseDate"]
            for row in data.unique("ContractCode")
            .select(["ContractCode", "ExerciseDate"])
            .iter_rows(named=True)
        }
        self.price_map = {
            (d, c): p
            for d, c, p in data.select(
                ["TradingDate", "ContractCode", "ClosePrice"]
            ).iter_rows()
        }
        self.atm_data = data.filter(pl.col("ATMFlag") == 1)

    def query_price(self, date, contract):
        return self.price_map[(date, contract)]

    def get_atm(self, date):
        data = (
            self.atm_data.filter(pl.col("TradingDate") == date)
            .filter(pl.col("ExerciseDate") > date + pl.duration(days=28))
            .sort("ExerciseDate")
            .group_by("CallOrPut")
            .head(1)
            .select("ContractCode", "CallOrPut", "ClosePrice", "Delta")
        )
        return data

    def get_value(self, date):
        val = 0
        for code, pos in self.position.items():
            price = self.query_price(date, code)
            value = price * pos["Volume"]
            val += value
        return val

    def get_cost(self):
        cost = 0
        for pos in self.position.values():
            price = pos["Cost"]
            value = price * pos["Volume"]
            cost += value
        return cost

    def get_profit(self, date):
        value = self.get_value(date)
        cost = self.get_cost()
        profit = (value - cost) / abs(cost) if cost != 0 else 0
        return profit

    def update_position(self, code, price, volume, cp):
        if code in self.position:
            old_vol = self.position[code]["Volume"]
            old_cost = self.position[code]["Cost"]

            if (old_vol > 0 and volume > 0) or (old_vol < 0 and volume < 0):
                value_p = old_cost * old_vol
                vol_tot = old_vol + volume
                self.position[code]["Cost"] = (value_p + price * volume) / vol_tot
                self.position[code]["Volume"] = vol_tot
            else:
                self.position[code]["Volume"] += volume
        else:
            self.position[code] = {
                "ContractCode": code,
                "CallOrPut": cp,
                "Volume": volume,
                "Cost": price,
            }

        fee = abs(volume) / 10000 * self.fee
        self.balance -= price * volume + fee

        if abs(self.position[code]["Volume"]) < 1e-5:
            del self.position[code]

    def open_position(self, date, side):
        # side: 1 for buy, -1 for sell
        dat = self.get_atm(date)
        if len(dat) < 2:
            return

        call_dat = dat.filter(pl.col("CallOrPut") == "C")
        put_dat = dat.filter(pl.col("CallOrPut") == "P")

        if call_dat.is_empty() or put_dat.is_empty():
            return

        price_c = call_dat["ClosePrice"][0]
        delta_c = call_dat["Delta"][0]
        code_c = call_dat["ContractCode"][0]

        price_p = put_dat["ClosePrice"][0]
        delta_p = put_dat["Delta"][0]
        code_p = put_dat["ContractCode"][0]

        ratio = abs(delta_c / (delta_p + 1e-5))

        pos_scale = self.para[{1: "position_long", -1: "position_short"}[side]]
        denom = price_c + ratio * price_p

        if denom == 0:
            return

        vol_c = (self.balance * pos_scale) / denom
        vol_c = int(vol_c / 10000) * 10000
        vol_p = int(vol_c * ratio / 10000) * 10000

        if vol_c > 0:
            self.update_position(code_c, price_c, vol_c * side, "C")
        if vol_p > 0:
            self.update_position(code_p, price_p, vol_p * side, "P")

    def clean_position(self, code, date):
        if code in self.position:
            price = self.query_price(date, code)
            volume = -self.position[code]["Volume"]
            cp = self.position[code]["CallOrPut"]
            self.update_position(code, price, volume, cp)

    def clean_all_position(self, date):
        for code in list(self.position.keys()):
            self.clean_position(code, date)

    def drop_close_position(self, date):
        to_delete = []
        for code in self.position.keys():
            exercise_date = self.exercise[code]
            if (exercise_date - date).days <= 15:
                to_delete.append(code)

        for code in to_delete:
            self.clean_position(code, date)

    def save_value(self, date):
        value = self.get_value(date)
        total_value = self.balance + value
        self.value.append(
            {"TradingDate": date, "TotalValue": total_value, "Position": value}
        )

    def get_factor_dict(self, date):
        factor = self.factors.filter(pl.col("TradingDate") == date)
        return list(factor.iter_rows(named=True))[0]

    def do_strategy(self, date):
        factor = self.get_factor_dict(date)

        # rank, term, skew, pcr
        s1 = factor["rank"] > self.para["threshold_rank"]
        s2 = factor["term"] > self.para["threshold_term"]
        s3 = factor["skew"] > self.para["threshold_skew"]
        s4 = factor["pcr"] > self.para["threshold_pcr"]

        l1 = factor["rank"] < 1 - self.para["threshold_rank"]
        l2 = factor["term"] < 1 - self.para["threshold_term"]
        l3 = factor["skew"] < 1 - self.para["threshold_skew"]
        l4 = factor["pcr"] < 1 - self.para["threshold_pcr"]

        score_l = int(l1) + int(l2) + int(l3) + int(l4)
        score_s = int(s1) + int(s2) + int(s3) + int(s4)

        condition_short = s1 and score_s > self.para["threshold_op"]
        condition_long = l1 and score_l > self.para["threshold_op"]

        self.drop_close_position(date)

        if self.position:
            profit = self.get_profit(date)
            is_long = list(self.position.values())[0]["Volume"] > 0
            cond = condition_long if is_long else condition_short

            if (
                profit < -self.para["threshold_profit"]
                or profit > self.para["threshold_profit"]
            ) and not cond:
                self.clean_all_position(date)
        else:
            if condition_long:
                self.open_position(date, 1)
            elif condition_short:
                self.open_position(date, -1)

        self.save_value(date)

    def run(self) -> pl.DataFrame:
        for date in self.factors.select("TradingDate").to_series():
            self.do_strategy(date)
        self.clean_all_position(date)

    def get_value_df(self) -> pl.DataFrame:
        return pl.DataFrame(self.value)


def get_param_grid():
    THRESHOLD1_VALUES = np.arange(0.75, 0.95, 0.05)
    THRESHOLD2_VALUES = np.arange(2, 4, 1)
    POSITION_VALUES = np.arange(0.10, 0.25, 0.05)
    PROFIT_VALUES = np.arange(0.10, 0.25, 0.05)

    para_list = itertools.product(
        itertools.product(THRESHOLD1_VALUES, repeat=4),
        THRESHOLD2_VALUES,
        itertools.product(POSITION_VALUES, repeat=2),
        PROFIT_VALUES,
    )

    param_grid = []
    for thresholds_1, thresholds_2, position, profit in para_list:
        param = {
            "threshold_rank": thresholds_1[0],
            "threshold_term": thresholds_1[1],
            "threshold_skew": thresholds_1[2],
            "threshold_pcr": thresholds_1[3],
            "threshold_op": thresholds_2,
            "threshold_profit": profit,
            "position_long": position[0],
            "position_short": position[1],
        }
        param_grid.append(param)

    return param_grid


def run_backtrace(
    data: pl.DataFrame, factors: pl.DataFrame, para: dict
) -> pl.DataFrame:
    backtrace = BackTrace(data, factors, balance=1_000_000, para=para)
    backtrace.run()
    result = backtrace.get_value_df()
    return result


def draw_backtrace_result(result: pl.DataFrame, label: str):
    plt.figure(figsize=(12, 6))
    plt.plot(
        result["TradingDate"],
        result["TotalValue"],
        label="Total Value",
        color="green",
    )
    plt.plot(
        result["TradingDate"],
        result["Position"],
        label="Position Value",
        color="blue",
    )
    plt.xlabel("Trading Date")
    plt.ylabel("Total Value")
    plt.title("Backtrace Result: Total Value Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"./output/backtrace_total_value_{label}.png")


def try_on_train(data: pl.DataFrame, factors: pl.DataFrame) -> dict:
    param_grid = get_param_grid()
    print(f"Total {len(param_grid)} parameter combinations to try.")
    res = []
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(run_backtrace)(data, factors, para) for para in param_grid
    )
    for r, para in zip(results, param_grid):
        returns = r.select("TotalValue").to_series().pct_change().mean() * 252
        std = r.select("TotalValue").to_series().pct_change().std() * np.sqrt(252)
        sharpe = returns / std if std != 0 else 0
        res.append((para, sharpe, r, returns, std))
    res.sort(key=lambda x: x[1], reverse=True)
    draw_backtrace_result(res[0][2], "train")
    print("Best returns on training:", res[0][3])
    print("Best Sharpe Ratio on training:", res[0][1])
    print("With parameters:", res[0][0])
    return res[0][0]


def try_on_test(data: pl.DataFrame, factors: pl.DataFrame, para: dict) -> pl.DataFrame:
    result = run_backtrace(data, factors, para)
    returns = result.select("TotalValue").to_series().pct_change().mean() * 252
    std = result.select("TotalValue").to_series().pct_change().std() * np.sqrt(252)
    sharpe = returns / std if std != 0 else 0
    print("Returns on testing:", returns)
    print("Sharpe Ratio on testing:", sharpe)
    draw_backtrace_result(result, "test")
