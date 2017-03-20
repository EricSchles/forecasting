import math
import datetime
import statistics

def date_to_datetime(time_string):
    return datetime.datetime.strptime(time_string, '%m/%d/%Y')

def is_nan(obj):
    if type(obj) == type(float()):
        return math.isnan(obj)
    else:
        return False
    
def money_to_float(string):
    """
    hourly wages have dollar signs and use commas, 
    this method removes those things, so we can treat stuff as floats
    """
    if type(string) == type(str()):
        string = string.replace("$","").replace(",","")
        return float(string)
    else:
        return string

def count_outliers(data):
    outliers_num = 0
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)
    for elem in data:
        if (elem > mean + (2*stdev)) or (elem < mean - (2*stdev)):
            outliers_num += 1
    return outliers_num

def first_quartile(List):
    if len(List) >= 4:
        List.sort()
        if len(List) % 2 != 0:
            middle_number = statistics.median(List)
            return statistics.median(List[:List.index(middle_number)])
        else:
            middle_index = len(List)//2
            middle_number = List[middle_index]
            return statistics.median(List[:List.index(middle_number)])
    else:
        return None    

def third_quartile(List):
    if len(List) >= 4:
        List.sort()
        if len(List) % 2 != 0:
            middle_number = statistics.median(List)
            return statistics.median(List[List.index(middle_number):])
        else:
            middle_index = len(List)//2
            middle_number = List[middle_index]
            return statistics.median(List[List.index(middle_number):])
    else:
        return None    

def quartile_deviation(data):
    q1 = first_quartile(data)
    q3 = third_quartile(data)
    median = statistics.median(data)
    if abs(median - q1) > abs(median - q3):
        return abs(median - q1)
    else:
        return abs(median - q3)

def min_max_range(data):
    data.sort()
    return abs(data[-1] - data[0])
