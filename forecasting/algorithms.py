
def _sliding_window(timeseries,sliding_window_size):
    """
    Input:
    * timeseries is a set of timeseries data, assumed to be evenly spaced
    * sliding_window_size is the size of the sliding window for the X_data that is returned
    Output:
    * two lists, one a set of sliding windows (X_data) and the other a list of output values (Y_data)
    assumptions:
    * timeseries data is the full set of data we want to prepare for timeseries forecasting
    * all date range components are removed at this point
    * all input and output data is assumed to be integers or floats
    """
    X_data = []
    Y_data = []
    #one of the sliding_window_size's is wrong
    tmp = timeseries[:sliding_window_size+1]
    X_data.append(tmp[:])
    #Y_data.append(
    for elem in timeseries[sliding_window_size+1:]:
        Y_data.append(elem)
        tmp.pop(0)
        tmp.append(elem)
        X_data.append(tmp[:])
    #X_data is chopped off at the end
    return X_data[:-1], Y_data 

def score_model(model,timeseries,sliding_window_size,percentage):
    #assume transforms come before this to ensure data quality
    x,y = _sliding_window(timeseries,sliding_window_size)
    threshold = int(percentage * len(x))
    x_train = x[:threshold]
    y_train = y[:threshold]
    x_test = x[threshold:]
    y_test = y[threshold:]
    result = model.fit(x_train,y_train)
    return result.score(x_test,y_test)

def forecast(forecast_periods, model, timeseries, sliding_window_size):
    #assume transforms come before this to ensure data quality
    x,y = _sliding_window(timeseries,sliding_window_size)
    result = model.fit(x,y)
    for i in range(forecast_periods):
        tmp = x[-1][:]
        prediction = list(result.predict([tmp]))[0]
        y.append(prediction)
        tmp.append(prediction)
        tmp.pop(0)
        x.append(tmp[:])
        result = model.fit(x,y)
    return x,y

