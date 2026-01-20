import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('coordinates/field5.csv')

# assume the label column is called 'label'
df0 = df[df['class'] == 0]
df1 = df[df['class'] == 1]

plt.figure()
plt.scatter(df0['longitude'], df0['latitude'], label='class 0')
plt.scatter(df1['longitude'], df1['latitude'], label='class 1')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend()
plt.show()
