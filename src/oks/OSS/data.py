import pandas as pd


# TODO: this function should be updated to return real-time data from APIS if the operator support systems is deployed
# The data format should be equal to the one in optimization_demo.csv unless the system is redesigned
def get_optimization_data(offset=0):
    data = get_historical_data("optimization_demo")
    prehistory = 60
    data_start_time = data.first_valid_index()
    data_end_time = data_start_time + pd.Timedelta(minutes=prehistory + offset)
    optimization_data = data[data_start_time:data_end_time]
    return optimization_data


def get_historical_data(filename):
    prefix = "src/oks/OSS/"
    suffix = ".csv"
    date_column = "Timestamp"
    data = pd.read_csv(
        prefix + filename + suffix,
        parse_dates=[date_column],
        index_col=date_column,
    )
    return data


