# Step 1: Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from xml.etree.ElementInclude import include
from sklearn.metrics import mean_squared_error

def FitScreen(z):
    x = [[0, 3], [0, 8], [0, 13], [0, 18],
         [5, 3], [5, 8], [5, 13], [5, 18],
         [10, 3], [10, 8], [10, 13], [10, 18],
         [15, 3], [15, 8], [15, 13], [15, 18],
         [20, 3], [20, 8], [20, 13], [20, 18],
         [25, 3], [25, 8], [25, 13], [25, 18],
         [30, 3], [30, 8], [30, 13], [30, 18]]
    
    y = [[-3, 0], [-3, 5], [-3, 16], [-3, 25],
         [4, 0], [4, 5], [4, 16], [4, 25],
         [11, 0], [11, 5], [11, 16], [11, 25],
         [16, 0], [16, 5], [16, 16], [16, 25],
         [22, 0], [22, 5], [22, 16], [22, 25],
         [29, 0], [29, 5], [29, 16], [29, 25],
         [35, 0], [35, 5], [35, 16], [35, 25]]
    
    x = np.array(x)
    y = np.array(y)
    
    polynomial_features= PolynomialFeatures(degree=4, include_bias = True)
    x_poly = polynomial_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly, y)
    z_trans = polynomial_features.fit_transform(z)
    
    #rmse_test = np.sqrt(mean_squared_error(y, model.predict(x_poly)))
    #print("rmse:", rmse_test)
    
    plt.plot(model.predict(polynomial_features.fit_transform(np.ones((100, 100)))))
    plt.show()
    return(model.predict(z_trans))

FitScreen([[1],[1]])