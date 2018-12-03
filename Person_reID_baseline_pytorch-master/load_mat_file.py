#give the filename as the input like get_data('market_attribute.mat')
#returns array 'res_arr' of size (1501,27) and corresponding labels can be found in the tuple 'attr_names'
import numpy as np
from scipy.io import loadmat

def get_data(filename):
    x=loadmat(filename)
    attribute_key = 'market_attribute'
    attr_names = x[attribute_key]['train'][0][0].dtype.names
    print(attr_names)
    res = []
    for index in range(len(x[attribute_key]['train'][0][0]['upgreen'][0][0][0])):
        temp = []
        for i in attr_names:
            temp.append(x[attribute_key]['train'][0][0][i][0][0][0][1])
        res.append(temp)

    for index in range(len(x[attribute_key]['test'][0][0]['upgreen'][0][0][0])):
        temp = []
        for i in attr_names:
            temp.append(x[attribute_key]['test'][0][0][i][0][0][0][1])
        res.append(temp)

    res_arr = np.array(res)
    return res_arr,attr_names
