import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)

from scipy import sparse

# create a 2d numpy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("Numpy array:\n%s" % eye)

# convert the numpy array to a scipy sparse matrix in CSR format
# only the non-zero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSR matrix:\n%s" % sparse_matrix)

import matplotlib.pyplot as plt

# Generate a sequence of integers
import pandas as pd

# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
    'Location' : ["New York", "Paris", "Berlin", "London"],
    'Age' : [24, 13, 53, 33]
    }

data_pandas = pd.DataFrame(data)
print(data_pandas)
