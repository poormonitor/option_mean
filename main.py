from read_data import read_combined
from draw_fig import draw_fig
from calc_factor import calculate_factor, divide_data
from do_backtrace import try_on_train, try_on_test


def main():
    data = read_combined()

    draw_fig(data)
    factors = calculate_factor(data)

    train_data, test_data, train_factors, test_factors = divide_data(
        data, factors, split_date="2023-01-01"
    )

    para = try_on_train(train_data, train_factors)
    try_on_test(test_data, test_factors, para)


if __name__ == "__main__":
    main()
