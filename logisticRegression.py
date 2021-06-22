import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('escape_velocity.csv')

velocity_list = df['Velocity'].tolist()

escape_list = df['Escaped'].tolist()

fig = px.scatter(x = velocity_list, y = escape_list)

fig.show()

velocity_array = np.array(velocity_list)

escape_array = np.array(escape_list)

m, c = np.polyfit(velocity_array, escape_array, 1)

y = []

for x in velocity_array:
  y_value = m * x + c
  y.append(y_value)

fig = px.scatter(x = velocity_array, y = escape_array)

fig.update_layout(shapes = [
                            dict(
                                type = 'line',
                                y0 = min(y), y1 = max(y),
                                x0 = min(velocity_array), x1 = max(velocity_array)
                            )
])

fig.show()

X = np.reshape(velocity_list, (len(velocity_list), 1))

Y = np.reshape(escape_list, (len(escape_list), 1))

lr = LogisticRegression()

lr.fit(X, Y)

plt.figure()

plt.scatter(X.ravel(), Y, color = 'black', zorder = 20)

def model(x):
  return 1/(1 + np.exp(-x))

X_test = np.linspace(0, 100, 200)

chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color = 'red', linewidth = 3)

plt.axhline(y = 0, color = 'k', linestyle = '-')

plt.axhline(y = 1, color = 'k', linestyle = '-')

plt.axhline(y = 0.5, color = 'b', linestyle = '--')

plt.axvline(x = X_test[165], color = 'b', linestyle = '--')

plt.ylabel('y')

plt.xlabel('X')

plt.xlim(0, 20)

plt.show()

user_score = float(input('Enter the velocity:'))

chances = model(user_score * lr.coef_ + lr.intercept_).ravel()[0]

if chances <= 0.01:
  print('Velocity will not get escaped.')
elif chances >= 1:
  print('Velocity will get escaped.')
elif chances < 0.5:
  print('Velocity might not get escaped.')
else:
  print('Velocity may get escaped.')