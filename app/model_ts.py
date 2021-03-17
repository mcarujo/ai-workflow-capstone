#!/usr/bin/env python
# coding: utf-8

import joblib
import numpy as np
import pandas as pd
from fbprophet import Prophet
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_data_ts():
    data = pd.read_csv("data/data_set_time_series.csv")

    data = data[data.Final_price > 0]
    data = data[data.Final_price < data.Final_price.quantile(0.98)]
    data.date = pd.to_datetime(data.date)
    full_dates = pd.DataFrame(
        pd.date_range(start=data.date.min(), end=data.date.max()), columns=["date"]
    )
    data = data.set_index("date").join(
        full_dates.set_index("date"), how='right')
    data["Final_price"].interpolate(method="linear", inplace=True)
    data["Final_times_viewed"].interpolate(method="linear", inplace=True)
    return data


def format_to_prophet(serie_ds, serie_y):
    aux = pd.DataFrame()
    aux["ds"] = serie_ds
    aux["y"] = serie_y
    return aux


def train_predict(
    data,
    periods,
    freq="W",
    train=False,
    yearly_seasonality=False,
    cps=1,
    changepoint_range=0.8,
):
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
        return model, forecast, future
    # Tuning
    else:
        return {"CPS": cps, "R2": r2, "MSE": mse, "MAE": mae}


def tunning_model(data):
    data_prophet = format_to_prophet(
        data.reset_index().date, data.reset_index().Final_times_viewed
    )
    cps_options = [round(x, 3)
                   for x in np.linspace(start=0.001, stop=5, num=50)]

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_predict)(
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
    model, forecast, future = train_predict(
        data=data_prophet,
        periods=30,
        freq="D",
        train=True,
        cps=results.iloc[0]["CPS"],
        yearly_seasonality=True,
    )
    return model


def save_joblib(model, path):
    joblib.dump(model, path)


if __name__ == "__main__":
    """
        Model training
    """
    data_ts = get_data_ts()
    model = tunning_model(data_ts)
    joblib.dump(model, 'model/prophet.joblib')
