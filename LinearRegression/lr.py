import numpy as np 
X_data = np.array([160, 171, 182, 180, 154], ndmin=2)
X_data = X_data.reshape((5,1))
y_data = np.array([72, 76, 77, 83, 76])
# print(X_data.shape, y_data)
X_mean = np.mean(X_data)
y_mean = np.mean(y_data)
n = len(X_data)

class LinearRegression:
    b0 = 0
    b1 = 0
    def LinearRegression_fit(self, X, y):
        upward_function = 0
        x_downward = 0
        for i in range(n):
            upward_function += (X_data[i]-X_mean)*(y_data[i]-y_mean)
            x_downward += (X_data[i]-X_mean)**2
        self.b1 = upward_function/x_downward
        self.b0 = y_mean - (self.b1*X_mean)
        return self.b0, self.b1

    def LinearRegression_predict(self, Xi):
        y_pred = self.b0 + (self.b1*Xi)
        return y_pred


model = LinearRegression()
print(model.LinearRegression_fit(X_data,y_data))
print(model.LinearRegression_predict(176))