from forecasting import helpers
import numpy as np
import random
import statistics

#testing helpers
def test_is_nan():
    assert helpers.is_nan(np.nan) == True
    assert helpers.is_nan("Hello") == False
    assert helpers.is_nan(5) == False
    assert helpers.is_nan(4.5) == False
    assert helpers.is_nan(False) == False
    assert helpers.is_nan(True) == False

def test_money_to_float():
    assert helpers.money_to_float(5.40) == 5.40
    assert helpers.money_to_float("5.40") == 5.40
    assert helpers.money_to_float("$5.50") == 5.50
    assert helpers.money_to_float("5,000.00") == 5000.00
    assert helpers.money_to_float("$5,000.00") == 5000.00

def test_min_max_range():
    test_data = [random.randint(0,100) for _ in range(10000)]
    assert helpers.min_max_range(test_data) == abs(max(test_data) - min(test_data)) 

def test_count_outliers_function():
    mu, sigma = 50, 3
    s = np.random.normal(mu, sigma, 10000)
    assert helpers.count_outliers(list(s)) < 10000 * 0.07 #the number of outliers is less than 7% of the data
    
def test_count_outlier_supporting_functions():
    mu, sigma = 50, 3
    s = np.random.normal(mu, sigma, 10000)
    assert round(statistics.mean(list(s)),5) == round(sum(list(s))/float(len(list(s))), 5)
    assert round(statistics.mean(list(s)),5) == round(float(np.mean(s)),5)
    assert round(statistics.stdev(list(s)),5) == round(float(np.std(s, ddof=1)),5)

def test_first_quartile():
    test_data = [random.randint(0,100) for _ in range(10000)]
    assert abs(helpers.first_quartile(test_data) - quartiles(test_data)[0]) < 2

def test_third_quartile():
    test_data = [random.randint(0,100) for _ in range(10000)]
    assert abs(helpers.third_quartile(test_data) - quartiles(test_data)[1]) < 2

def test_quartile_deviation():
    test_data = [random.randint(0,100) for _ in range(10000)]
    value = max(abs(statistics.median(test_data)- quartiles(test_data)[1]), abs(statistics.median(test_data)-quartiles(test_data)[0]))
    assert abs(helpers.quartile_deviation(test_data) - value) < 2

def test_first_third_quartile_supporting_functions():
    mu, sigma = 50, 3
    s = np.random.normal(mu, sigma, 10000)
    assert round(statistics.median(list(s)),5) == round(float(np.median(s)),5)
    
    
#curtosy of https://github.com/Mashimo/datascience/blob/master/datascience/stats.py
def quartiles(dataPoints):
    """
    the lower and upper quartile
    Arguments:
        dataPoints: a list of data points, int or float
    Returns:
        the first and the last quarter in the sorted list, a tuple of float or int
    """
    if not dataPoints:
        raise StatsError('no data points passed')
        
    sortedPoints = sorted(dataPoints)
    mid = len(sortedPoints) // 2 # uses the floor division to have integer returned
    
    if (len(sortedPoints) % 2 == 0):
        # even
        lowerQ = median(sortedPoints[:mid])
        upperQ = median(sortedPoints[mid:])
    else:
        # odd
        lowerQ = median(sortedPoints[:mid])
        upperQ = median(sortedPoints[mid+1:])
            
    return (lowerQ, upperQ)

def median(dataPoints):
    """
    the median of given data
    Arguments:
        dataPoints: a list of data points, int or float
    Returns:
        the middle number in the sorted list, a float or an int
    """
    if not dataPoints:
        raise StatsError('no data points passed')
        
    sortedPoints = sorted(dataPoints)
    mid = len(sortedPoints) // 2  # uses the floor division to have integer returned
    if (len(sortedPoints) % 2 == 0):
        # even
        return (sortedPoints[mid-1] + sortedPoints[mid]) / 2.0
    else:
        # odd
        return sortedPoints[mid]

