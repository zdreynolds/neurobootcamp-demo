import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv("../data/raw_data.csv")
    denoise_signal(df, "temperature")

def denoise_signal(df, varname):
    signal        = df[varname].dropna()
    lagged_signal = build_lags(signal)
    y_hat         = fit_model_and_predict(lagged_signal)
    make_plots(signal, y_hat, varname)
    

def build_lags(signal):

    # Create 3rd order lag model dataframe
    lagged_signal = pd.DataFrame({
        'y': signal,
        'y_lag_1': signal.shift(1),
        'y_lag_2': signal.shift(2),
        'y_lag_3': signal.shift(3)
    }).dropna()

    return lagged_signal

def fit_model_and_predict(lagged_signal):
    X = lagged_signal[['y_lag_1', 'y_lag_2', 'y_lag_3']]
    y = lagged_signal['y']

    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)

    return y_hat

def make_plots(signal, y_hat, varname):
    # Align y_hat with original series (accounting for dropped observations)
    y_hat_aligned = pd.Series(index=signal.index, dtype=float)
    y_hat_aligned.iloc[3:] = y_hat  # Start from index 3 since we dropped first 3 obs

    plt.plot(signal, label='Original')
    plt.plot(y_hat_aligned, label='De-noised')
    plt.legend()
    plt.savefig(f'../plots/{varname}_denoised.svg')
    plt.close()

if __name__ == "__main__":
    main()
