# Import Necessary Packages
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
import math
import seaborn as sns
from matplotlib.pylab import rcParams
plt.style.use('fivethirtyeight')
from fbprophet import Prophet as proph
import numpy as np



def extract_stats(forecast, zipcode, df):
    """
    Adds pertinent statistics from zipcode's forecast at 60 months (2023-04-01) 
    to the passed datafame.
    
    Args:
    df: dataframe with columns: zipcode, 2018-04-01, minimum, min_roi, expected, 
    expected_roi, maximum, max_roi. minimum, expected, maximum refer to 2023-04-01.
    zipcode: zipcode
    forecast: forecast from model_price or fbprohet.Prophet
    
    Returns:
    the dataframe
    """
    current = forecast.loc[forecast['ds']=='2018-04-01', 'yhat'].values[0]
    minimum = forecast.loc[forecast['ds']=='2023-03-01', 'yhat_lower'].values[0]
    expected = forecast.loc[forecast['ds']=='2023-03-01', 'yhat'].values[0]
    maximum = forecast.loc[forecast['ds']=='2023-03-01', 'yhat_upper'].values[0]
    min_roi = calc_roi(current, minimum)
    expected_roi = calc_roi(current, expected)
    max_roi = calc_roi(current, maximum)
    zip_df = pd.DataFrame([zipcode, current, minimum, min_roi, expected,
                             expected_roi, maximum, max_roi]).T
    zip_df.columns = ['zipcode', '2018-04-01', 'minimum', 'min_roi', 'expected', 
                      'expected_roi', 'maximum', 'max_roi']
    df = pd.concat([df, zip_df])
    return df



def calc_roi (initial_price, projected_price):
    """
    Calculates average ROI for a zipcode using the initial 
    price and projected price.
    """

    return (projected_price - initial_price) / initial_price




def plot_model(df, zipcode):
    """
    Takes a data frame and zipcode to create a plot showing average housing
    prices and forcasted prices from April 2005 to April 20018
    """

    model, forecast, zipcode = model_price(df, zipcode, '2005-04-01', '2018-04-01')
    model.plot(forecast, uncertainty=True)
    plt.title(zipcode)
    plt.show()




def melt_data(df):
    """
    Transposes a data frame from wide to long and groups by time
    """

    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'City', 'State',
                         'Metro', 'CountyName', 'SizeRank'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})



def model_price(df, zipcode, start, end, periods=60, freq='MS',
                 interval_width=0.80):
    """
    Takes in the data frame, zipcode, start and end dates, and a projection
    period length and returns a model, forecast, and zipcode.
    """
    df = melt_data(df.loc[df['RegionName'] == zipcode])
    df = df.reset_index()
    df.columns = ['ds', 'y']
    data = df.loc[(df['ds'] > start) & (df['ds'] < end)]
    Model = proph(interval_width=interval_width)
    Model.fit(data)
    future_dates = Model.make_future_dataframe(periods=periods, freq=freq)
    forecast = Model.predict(future_dates)
    return Model, forecast, zipcode


def model_extract_all(all_df):
    """
    iterates over each zipcode (RegionName), return a dataframe with the useful variables
    from modeling.
    
    Args:
        all_df: a data frame with columns RegionID, RegionName, City, State, Metro,
        CountyName, SizeRank, 1996-04, 1996-05, 1996-06... 2018-02, 2018-03, 2018-04
    Returns:
        stats_df: a dataframe with columns ['zipcode', '2018-04-01', 'minimum', 'min_roi', 
        'expected', 'expected_roi', 'maximum', 'max_roi']

        *** Run time was 17 hours to build the new dataframe**
        -stats.csv in raw_data has the results for easier use.
        """
    stats_df = pd.DataFrame(columns=['zipcode', '2018-04-01', 'minimum', 'min_roi', 'expected', 
                          'expected_roi', 'maximum', 'max_roi'])
    for index, zipcode in enumerate(all_df['RegionName'].values):
        model, forecast, zipcode = model_price(all_df, zipcode, '2005-04-01', '2018-04-01')
        stats_df = extract_stats(forecast, zipcode, stats_df)
        print(index)
    return stats_df