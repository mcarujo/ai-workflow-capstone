#!/usr/bin/env python
# coding: utf-8

import joblib
import os
import numpy as np
import pandas as pd
from fbprophet import Prophet
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrain:
    def __init__(self, dataset):
        self.dataset = dataset

    def process_dataset(self):
        """
            Load the time series from a csv then Impute, Transform and Clean.
        """
        dataset = self.dataset
        dataset = dataset[dataset.Final_price > 0]
        dataset = dataset[dataset.Final_price <
                          dataset.Final_price.quantile(0.98)]
        dataset.date = pd.to_datetime(dataset.date)
        full_dates = pd.DataFrame(
            pd.date_range(start=dataset.date.min(), end=dataset.date.max()), columns=["date"]
        )
        dataset = dataset.set_index("date").join(
            full_dates.set_index("date"), how='right')
        dataset["Final_price"].interpolate(method="linear", inplace=True)
        dataset["Final_times_viewed"].interpolate(
            method="linear", inplace=True)
        return dataset

    def format_to_prophet(self, serie_ds, serie_y):
        """
            Adapt the Time Series DataFrame to Prophet DataFrame.
        """
        aux = pd.DataFrame()
        aux["ds"] = serie_ds
        aux["y"] = serie_y
        return aux

    def train_predict(self,
                      data,
                      periods,
                      freq="W",
                      train=False,
                      yearly_seasonality=False,
                      cps=1,
                      changepoint_range=0.8,
                      ):
        """
            This function will be responsible to get the data and the model parameters, train and then return the metrics for evaluation.
        """
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=cps,
        )
        model.fit(data[:-periods])

        future = model.make_future_dataframe(
            periods=periods, freq=freq, include_history=True
        )
        forecast = model.predict(future)

        r2 = round(r2_score(data["y"], forecast["yhat"]), 3)
        mse = round(mean_squared_error(data["y"], forecast["yhat"]), 3)
        mae = round(mean_absolute_error(data["y"], forecast["yhat"]), 3)

        # Only train
        if train:
            print("R2: ", r2)
            print("MSE: ", mse)
            print("MAE: ", mae)
            return model, [r2, mse, mae]
        # Tuning
        else:
            return {"CPS": cps, "R2": r2, "MSE": mse, "MAE": mae}

    def tunning_model(self, data):
        """
            This is a Tunning Model will get the data, tuning the model and then train the model with the best parameters. 
        """
        data_prophet = self.format_to_prophet(
            data.reset_index().date, data.reset_index().Final_times_viewed
        )
        cps_options = [round(x, 3)
                       for x in np.linspace(start=0.001, stop=5, num=50)]

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(self.train_predict)(
                data=data_prophet,
                periods=30,
                freq="D",
                train=False,
                cps=i,
                yearly_seasonality=True,
            )
            for i in cps_options
        )

        results = pd.DataFrame(results)
        results = results[results.R2.isin([max(results.R2)])]
        results = results[results.MSE.isin([min(results.MSE)])]
        return self.train_predict(
            data=data_prophet,
            periods=30,
            freq="D",
            train=True,
            cps=results.iloc[0]["CPS"],
            yearly_seasonality=True,
        )

    def save_joblib(self, model, path):
        joblib.dump(model, path)

    def run(self):
        data_ts = self.process_dataset()
        model, metrics = self.tunning_model(data_ts)
        self.save_joblib(model, 'model/prophet.joblib')
        return model, metrics


class ModelPredict:
    def __init__(self):
        self.model = joblib.load(os.path.join('model', 'prophet.joblib'))

    def load_joblib(self, path):
        return joblib.load(path)

    def predict(self, days):
        future = self.model.make_future_dataframe(
            periods=days, freq='D', include_history=False
        )
        return self.model.predict(future)[['ds', 'trend', 'yhat_lower', 'yhat_upper', 'yhat']]


if __name__ == "__main__":
    """
        Model training test.
    """
    test = ModelPredict()
    print(test.predict(10)['ds', 'trend',
                           'yhat_lower', 'yhat_upper', 'yhat'])
