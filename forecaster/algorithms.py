import statistics as stat
from sklearn import svm, linear_model
import pandas as pd
import random

def moving_median(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_median_grouped(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median_grouped(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_median_high(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median_high(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_median_low(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median_low(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_average(df, sliding_window=1):
    means = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        means.append(stat.mean(data[start:end]))
        start += 1
        end += 1
    return means

def fit(df, sliding_window=1):
    """
    We assume that a single column is passed in

    """
    df = pd.DataFrame()
    for _ in range(1000):
        df = df.append({"a":random.randint(0,100)}, ignore_index=True)
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind][df.columns[0]] for ind in df.index]
    Xs = []
    Ys = []
    while x_end != len(data)-1:
        Ys.append(data[x_end+1])
        Xs.append(data[x_start:x_end])
        x_start += 1
        x_end += 1
    regressor = svm.SVR()
    regressor.fit(Xs,Ys)
    return regressor

def predict(regressor):
    
    
def svm_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = svm.SVR()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
        x_start += 1
        x_end += 1
    return predictions

def ridge_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.Ridge()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def lasso_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.Lasso()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def elastic_net_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.ElasticNet()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def orthogonal_matching_pursuit_cv_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.OrthogonalMatchingPursuitCV()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def bayesian_ridge_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.BayesianRidge()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def automatic_relevance_determination_regression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.ARDRegression()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions


def model_search(data):
    """
    Optimizers supported:
    * brute
    """
    print("got to start of model search")
    results = []
    results.append(brute_search(data))
    min_score = 100000000
    best_order = ()
    for result in results:
        if result[1] < min_score:
            min_score = result[1]
            best_order = result[0]
    return best_order


def date_to_datetime(time_string):
    return datetime.datetime.strptime(time_string, '%m/%d/%Y')


def ave_error(vals, f_vals):
    """
    parameters:
    @vals - the values observed
    @f_vals - the fitted values from the model
    """
    if (len(vals) == len(f_vals)) or (len(vals) < len(f_vals)):
        ave_errors = [abs(vals[ind] - f_vals[ind]) for ind in range(len(vals))]
        return sum(ave_errors)/len(vals)
    else:
        ave_errors = [abs(vals[i] - f_vals[i]) for i in range(len(f_vals))]
        return sum(ave_errors)/len(vals)


def make_prediction(model):
    df = pd.DataFrame()
    number_observations = len(model.fittedvalues)
    date_list = [i.to_datetime() for i in list(model.fittedvalues.index)]
    
    if number_observations >= 100:
        start = int(number_observations / 2)
        deltas = []
        for index in range(len(date_list)-1):
            deltas.append(date_list[index+1] - date_list[index])
        time_difference_in_days = [delta.days for delta in deltas]
        average_delta_days = statistics.mean(time_difference_in_days)
        stdev_delta_days = statistics.stdev(time_difference_in_days)
        median_delta_days = statistics.median(time_difference_in_days)
        total_days_in_5_years = 1825
        if stdev_delta_days < average_delta_days or median_delta_days <= 0.5:
            end = number_observations + int(total_days_in_5_years/average_delta_days)
        else:
            end = number_observations + int(total_days_in_5_years/median_delta_days)
    else:
        start = 1
        end = number_observations + 100
    #this is the method I need - model.forecast
    #result = model.forecast(start = start, end = end, dynamic = True)
    #model.plot_predict(start, end, dynamic = True)
    #plt.show()
    prediction = model.predict(start=start, end=end, dynamic=True)
    forecasted = model.forecast(steps=60)
    return prediction, forecasted
    
# create a monthly continuous time series to pull from
# interpolate values from trend
# set prediction from new values from artificially generated time series.

def setting_y_axis_intercept(data, interpolated_data, model):
    try:
        # if we are using the original data
        data = list(interpolated_data["Price"])
    except:
        # if we are using the deseasonalized data
        data = list(interpolated_data)
    fittedvalues_with_prediction = make_prediction(model)
    fittedvalues = model.fittedvalues
    avg = statistics.mean(data)
    median = statistics.median(data)
    possible_fitted_values = []
    possible_predicted_values = []
    
    possible_fitted_values.append([elem + avg for elem in fittedvalues])
    possible_fitted_values.append([elem + data[0] for elem in fittedvalues])
    possible_fitted_values.append([elem + median for elem in fittedvalues])
    possible_fitted_values.append(fittedvalues)
    possible_predicted_values.append([elem + avg for elem in fittedvalues])
    possible_predicted_values.append([elem + data[0] for elem in fittedvalues])
    possible_predicted_values.append([elem + median for elem in fittedvalues])
    possible_predicted_values.append(fittedvalues)

    min_error = 1000000
    best_fitted_values = 0
    for ind, f_values in enumerate(possible_fitted_values):
        avg_error = ave_error(data, f_values)
        if avg_error < min_error:
            min_error = avg_error
            best_fitted_values = ind
    print("minimum error:", min_error)
    return possible_predicted_values[best_fitted_values]


def calculate_error(data, model):
    try:
        # if we are using the original data
        data = list(data["Price"])
    except:
        # if we are using the deseasonalized data
        data = list(data)
    fittedvalues = model.fittedvalues
    avg = statistics.mean(data)
    median = statistics.median(data)
    possible_fitted_values = []
    possible_fitted_values.append([elem + avg for elem in fittedvalues])
    possible_fitted_values.append([elem + data[0] for elem in fittedvalues])
    possible_fitted_values.append([elem + median for elem in fittedvalues])
    possible_fitted_values.append(fittedvalues)

    min_error = 1000000
    best_fitted_values = 0
    for ind, f_values in enumerate(possible_fitted_values):
        avg_error = ave_error(data, f_values)
        if avg_error < min_error:
            min_error = avg_error
            best_fitted_values = ind
    return min_error/len(data)


def check_for_extreme_values(sequence, sequence_to_check=None):
    mean = statistics.mean(sequence)
    stdev = statistics.stdev(sequence)
    if sequence_to_check is not None:
        for val in sequence_to_check:
            if val >= mean + (stdev*2):
                sequence_to_check.remove(val)
            elif val <= mean - (stdev*2):
                sequence_to_check.remove(val)
            return sequence_to_check
    else:
        for val in sequence:
            if val >= mean + (stdev*2):
                sequence.remove(val)
            elif val <= mean - (stdev*2):
                sequence.remove(val)
        return sequence

    
def clean_data(data):
    new_data = pd.DataFrame()
    for timestamp in set(data.index):
        if len(data.ix[timestamp]) > 1:
            tmp_df = data.ix[timestamp].copy()
            new_price = statistics.median([tmp_df.iloc[index]["Price"] for index in range(len(tmp_df))])
            series = tmp_df.iloc[0]
            series["Price"] = new_price
            new_data = new_data.append(series)
        else:
            new_data = new_data.append(data.ix[timestamp])
    return new_data
