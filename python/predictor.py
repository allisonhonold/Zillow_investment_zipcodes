# Import Necessary Packages
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
from fbprophet import Prophet



def extract_stats(forecast, zipcode, df):
    """Adds pertinent statistics from zipcode's forecast at 60 months (2023-04-01) 
    to the passed datafame.
    
    Args:
    df: dataframe with columns: zipcode, 2018-04-01, minimum, min_roi, expected, 
    expected_roi, maximum, max_roi. minimum, expected, 
    maximum refer to 2023-04-01.
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
    """Calculates average ROI for a zipcode using the initial 
    price and projected price.

    Args:
    Returns:
    """

    return (projected_price - initial_price) / initial_price


def melt_data(df):
    """Transposes a data frame from wide to long and groups by time

    Args:
    Returns:
    """

    melted = pd.melt(df, id_vars=['RegionName', 'RegionID', 'City', 'State',
                         'Metro', 'CountyName', 'SizeRank'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})


def model_price(df, zipcode, start, end, periods=60, freq='MS',
                 interval_width=0.80):
    """Models the median price of homes in the zipcode.
    
    Args:
        df: data frame of historical housing price data
        zipcode: zipcode (int)
        start: start date for analysis
        end: end date for analysis
        periods: number of periods for projection
        freq: frequency of projection
        interval_width: confidence interval of projected max and min
    Returns:
        Model: fbprophet Model object
        forecast: a dataframe with modeled data
        zipcode: zipcode
    """
    historic_data = create_zip_df(df, zipcode, start, end)
    Model = Prophet(interval_width=interval_width, weekly_seasonality=False,
                    daily_seasonality=False)
    Model.fit(historic_data)
    future_dates = Model.make_future_dataframe(periods=periods, freq=freq)
    forecast = Model.predict(future_dates)
    return Model, historic_data, forecast, zipcode


def model_extract_all(all_df):
    """Iterates over each zipcode (RegionName), return a dataframe with our 
    target variables.

    *** Run time was 17 hours to build the new dataframe**
    /stats.csv in raw_data has the results for easier use.
    
    Args:
        all_df: a data frame with columns RegionID, RegionName, City, State, 
        Metro, CountyName, SizeRank, 1996-04, 1996-05, 1996-06... 2018-02, 
        2018-03, 2018-04
    Returns:
        stats_df: a dataframe with columns ['zipcode', '2018-04-01', 'minimum',
        'min_roi', 'expected', 'expected_roi', 'maximum', 'max_roi']
    """
    stats_df = pd.DataFrame(columns=['zipcode', '2018-04-01', 'minimum', 
                                     'min_roi', 'expected', 
                                     'expected_roi', 'maximum', 'max_roi'])
    for index, zipcode in enumerate(all_df['RegionName'].values):
        _, _, forecast, zipcode = model_price(all_df, zipcode, '2005-04-01', 
                                               '2018-04-01')
        stats_df = extract_stats(forecast, zipcode, stats_df)
        print(index)
    return stats_df


def create_zip_df(df, zipcode, start, end):
    """Creates a long dataframe in the format required by fbProphet for a 
    single zipcode with data points from the specified data range.

    Args:
        df: wide dataframe with one row per zipcode
        zipcode: zipcode (int)
        start: start date YYYY-MM-01
        end: end date YYYY-MM-01
    Returns:
        data_df: long dataframe with columns ds and y
    """
    df = melt_data(df.loc[df['RegionName'] == zipcode])
    df = df.reset_index()
    df.columns = ['ds', 'y']
    data_df = df.loc[(df['ds'] > start) & (df['ds'] < end)]
    return data_df


def plot_models(df, stats_df, zip_forecasts, title):
    """ Plots one or multiple models on a single axis. 
    
    Args: 
        df: a data frame with columns [RegionID, RegionName, City, State, Metro,
        CountyName, SizeRank, 1996-04, 1996-05, 1996-06... 2018-02, 2018-03, 2018-04]
        stats_df: a data frame with columns [zipcode, 2018-04-01, minimum, min_roi, 
        expected, expected_roi, maximum, max_roi]
        zip_forecasts: a dictionary of dictionaries in format 
        {zipcode: {history: history_df, forecast: forecast_df}}
    """
    for key in zip_forecasts.keys():
        assert isinstance(key, float), 'Zipcodes must be floats'

    palette = plt.get_cmap('Dark2')
    fig, ax = plt.subplots(figsize=(15,8))
    for index, zip_forecast in enumerate(zip_forecasts):
        color = palette(index)
        zipcode = zip_forecast
        history_df = zip_forecasts[zipcode]['history']
        forecast_df = zip_forecasts[zipcode]['forecast']
        zip_data = df.loc[df['RegionName'] == zipcode]
        plot_model(zip_data['City'].values[0], zip_data['State'].values[0],
                   history_df, forecast_df, color, ax)
    fig.suptitle(f'Recommended Zipcodes for Real Estate Investment: {title}')
    ax.legend(loc='upper left')
    ax.set_ylabel('price')
    ax.set_xlabel('year')


def plot_model(city, state, history_df, forecast_df, color, ax):
    """Plots a single model

    Args:
        city: city name (string)
        state: state (string)
        history_df: dataframe with historic data you want plotted
        forecast_df: dataframe with forecasted data you want plotted
        color: color for plot
        ax: axis for plotting
    """
    # plot historic data
    ax.plot(history_df['ds'], history_df['y'], color='k', alpha=0.5) 
    
    # plot forecast
    ax.plot_date(forecast_df['ds'], forecast_df['yhat'], color=color, alpha=.75,
                 label=f"{city}, {state}")

    # plot forecast range
    ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'],
                    forecast_df['yhat_upper'],
                    color=color, alpha=.25)


def plot_model_vs_real(zip_forecasts, true_vals, date):
    """Plots the forecasts and true value for each zipcode passed
    
    Args:
        zipcodes: an iterable of zipcodes
        forecasts: a dictionary of dictionaries 
            {zipcode: {history: history_df, forecast: forecast_df}}
        true_vals: a data frame with columns zipcode, price
        date: date for forecasting in format YYYY-MM-01
    
    Returns:
    """
    _, ax = plt.subplots()
    for zip_forecast in zip_forecasts:
        zipcode = zip_forecast
        forecast = zip_forecasts[zipcode]['forecast']
        zip_label = str(int(zipcode))
        ax.plot([zip_label, zip_label], 
                [forecast.loc[forecast['ds'] == date, 'yhat_lower'].values[0], 
                 forecast.loc[forecast['ds'] == date, 'yhat_upper'].values[0]],
                color = 'b', alpha = .5
               )
        ax.scatter(zip_label, 
                   forecast.loc[forecast['ds'] == date, 'yhat'].values[0],
                   color = 'b'
                   )
        ax.scatter([zip_label], 
                   [true_vals.loc[true_vals['zipcode'] == zipcode,'price'].values[0]],
                   color = 'k', 
                   marker = 'x'
                   )
    ax.set_ylim((0, 900000))
    ax.set_title(f"{date} Snapshot of Forecasted and Real Values")
    ax.set_ylabel('median home price')
    ax.set_xlabel('zipcode')