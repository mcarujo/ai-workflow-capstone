#!/usr/bin/env python
# coding: utf-8

import logging
import os
import re

import numpy as np
import pandas as pd


# Geting
class DataProcessing:
    def __init__(self):
        """
            constructor setting some local variables.
        """
        self.unused_columns = ["StreamID", "stream_id", "customer_id", "total_price",
                               "TimesViewed", "price", "times_viewed", "year", "month", "day"]
        self.DATA_DIR = os.path.join(
            "..",  "solution-guidance", "cs-train")
        self.DATA_OUT = os.path.join('data')

    def load_all_json_by_dir(self):
        """
            Load all files, concat them and then transform as DataFrame.
        """
        logging.info('Loading all json"s...')
        dfs = list()
        files = os.listdir(self.DATA_DIR)
        for file in files:
            logging.info(f'Reading the file: {file}')
            dfs.append(pd.read_json(os.path.join(
                self.DATA_DIR, file), orient="records"))
        return pd.concat(dfs).fillna(np.nan)

    def dropping_unnescessary_columns(self, dataframe):
        """
            Droping unnescessary columns in the DataFrame.
        """
        logging.info('Dropping columns.')
        return dataframe.drop(self.unused_columns, axis=1)

    def remove_non_numerical(self, string):
        """
            Using Regex to replace non numerical values by empty char "".
        """
        return re.sub("[^0-9]", "", string)

    def transforming_columns(self, dataframe):
        """
            Feature Engeering step, transforming cleaning features.
        """
        logging.info('Transforming features.')
        total_price_cleaned = dataframe[
            (dataframe["total_price"] <
             dataframe["total_price"].quantile(0.99))
            & (dataframe["total_price"] > 0)
        ]["total_price"].dropna()
        price_cleaned = dataframe[
            (dataframe["price"] < dataframe["price"].quantile(
                0.99)) & (dataframe["price"] > 0)
        ]["price"].dropna()
        times_viewed_cleaned = dataframe[
            (dataframe["times_viewed"] <
             dataframe["times_viewed"].quantile(0.99))
            & (dataframe["times_viewed"] > 0)
        ]["times_viewed"].dropna()
        TimesViewed_cleaned = dataframe[
            (dataframe["TimesViewed"] <
             dataframe["TimesViewed"].quantile(0.99))
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
        dataframe.invoice = dataframe.invoice.apply(self.remove_non_numerical)
        return dataframe

    def create_timeseries(self, dataframe):
        """
            Transform our dataset in a time series.
        """
        logging.info('Creating time series.')
        time_serie = dataframe.groupby("date").sum().reset_index()
        return time_serie

    def get_dataframe_to_train(self):
        # Data Processing
        logging.info('Geting datafram to train.')
        data_set = self.load_all_json_by_dir()
        data_set_transformed = self.transforming_columns(data_set)
        data_set_cleaned = self.dropping_unnescessary_columns(
            data_set_transformed)
        data_set_time_series = self.create_timeseries(data_set_cleaned)

        # Saving Dataset Regression
        logging.info('Saving Dataset Regression.')
        data_set_cleaned.to_csv(os.path.join(
            self.DATA_OUT, 'data_set.csv'), index=False)

        # Saving Dataset Time Series
        logging.info('Saving Dataset Time Series.')
        data_set_time_series.to_csv(os.path.join(
            self.DATA_OUT, 'data_set_time_series.csv'), index=False)

        return data_set_time_series
