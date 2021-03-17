#!/usr/bin/env python
# coding: utf-8

import os
import re

import numpy as np
import pandas as pd


# Geting
def load_all_json_by_dir(data_dir):
    """
        Load all files, concat them and then transform as DataFrame.
    """
    print('Loading all json"s...')
    dfs = list()
    files = os.listdir(data_dir)
    for file in files:
        print('Reading the file:', file)
        dfs.append(pd.read_json(os.path.join(
            data_dir, file), orient="records"))
    return pd.concat(dfs).fillna(np.nan)


def dropping_unnescessary_columns(dataframe):
    """
        Droping unnescessary columns in the DataFrame.
    """
    print('Dropping columns...')
    return dataframe.drop(["StreamID", "stream_id", "customer_id", "total_price", "TimesViewed", "price", "times_viewed", "year", "month", "day"], axis=1)


def remove_non_numerical(string):
    """
        Using Regex to replace non numerical values by empty char "".
    """
    return re.sub("[^0-9]", "", string)


def transforming_columns(dataframe):
    """
        Feature Engeering step, transforming cleaning features.
    """
    print('Transforming features...')
    total_price_cleaned = dataframe[
        (dataframe["total_price"] < dataframe["total_price"].quantile(0.99))
        & (dataframe["total_price"] > 0)
    ]["total_price"].dropna()
    price_cleaned = dataframe[
        (dataframe["price"] < dataframe["price"].quantile(
            0.99)) & (dataframe["price"] > 0)
    ]["price"].dropna()
    times_viewed_cleaned = dataframe[
        (dataframe["times_viewed"] < dataframe["times_viewed"].quantile(0.99))
        & (dataframe["times_viewed"] > 0)
    ]["times_viewed"].dropna()
    TimesViewed_cleaned = dataframe[
        (dataframe["TimesViewed"] < dataframe["TimesViewed"].quantile(0.99))
        & (dataframe["TimesViewed"] > 0)
    ]["TimesViewed"].dropna()

    dataframe["Final_times_viewed"] = dataframe["times_viewed"].fillna(
        0.0) + dataframe["TimesViewed"].fillna(0.0)
    dataframe["Final_price"] = dataframe["total_price"].fillna(
        0.0) + dataframe["price"].fillna(0.0)
    dataframe["date"] = dataframe[["year", "month", "day"]].apply(
        lambda row: "-".join(row.values.astype(str)), axis=1
    )
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe.invoice = dataframe.invoice.apply(remove_non_numerical)
    return dataframe


def create_timeseries(dataframe):
    """
        Transform our dataset in a time series.
    """
    print('Creating time series...')
    time_serie = dataframe.groupby("date").sum().reset_index()
    return time_serie


if __name__ == "__main__":
    """
        Data Processing flow
    """
    DATA_DIR = "../solution-guidance/cs-train"
    DATA_OUT = 'data'
    # Data Processing
    data_set = load_all_json_by_dir(DATA_DIR)
    data_set_transformed = transforming_columns(data_set)
    data_set_cleaned = dropping_unnescessary_columns(data_set_transformed)
    data_set_time_series = create_timeseries(data_set_cleaned)

    # Saving Dataset Regression
    data_set_cleaned.to_csv(os.path.join(
        DATA_OUT, 'data_set.csv'), index=False)

    # Saving Dataset Time Series
    data_set_time_series.to_csv(os.path.join(
        DATA_OUT, 'data_set_time_series.csv'), index=False)
