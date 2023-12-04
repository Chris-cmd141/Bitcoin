## import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer




##import libraries for ANN
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

## warning
import warnings
warnings.filterwarnings('ignore')

## load my dataset
df_original = pd.read_csv('Fear_Greed_Bitcoin.csv')
df_original.head()
# Convert the "Date" column to datetime
##df_original['Date'] = pd.to_datetime(df_original['Date'], format='%d/%m/%Y')
## FILLING THE MISSING VALUES FOR "Fear_Index" COLUMN
missing_values = df_original['Fear_Index'].median()
df_original['Fear_Index'].fillna(missing_values, inplace=True)
## drop "Value_Classification" column
df_original=df_original.drop(columns=['Value_Classification'])
df_original


df_mortgage = pd.read_csv('Mortgage_Rates_US.csv')
df_mortgage.head()
## Convert the "Date" column to datetime
##df_mortgage['Date'] = pd.to_datetime(df_mortgage['Date'], format='%d/%m/%Y')
# Fill missing mortgage values with the previous available value (forward-fill)
df_mortgage


df_nasdaq = pd.read_csv('Nasdaq.csv')
df_nasdaq.head()
## Convert the "Date" column to datetime
df_nasdaq

# Merge the DataFrames on the "Date" column to match mortgage values with corresponding dates
df_1 = df_original.merge(df_mortgage, on='Date', how='left')
df_1
df = df_1.merge(df_nasdaq, on='Date', how='left')
df



df['Mortgage_Rate_US'].fillna(method='ffill', inplace=True)
df
df['Nasdaq_close'].fillna(method='ffill', inplace=True)
df
df['Nasdaq_Volume'].fillna(method='ffill', inplace=True)
df

df.isnull().sum()

df.head(10)
df.to_csv('merged_data.csv', index=False)



df.info()
df_matrix = df.drop(columns=['Date'])
pd.set_option('display.float_format', '{:.2f}'.format) ## telling Pandas to format floating-point numbers with two decimal places
plt.figure(figsize=(8,5),dpi=200)
cmap = sns.color_palette("Blues")
sns.heatmap(df_matrix.corr(), annot = True, cmap=cmap)
df_matrix.corr()

df.info()

## Relation between DFF% and BTC_price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='DFF%', y='BTC_price', data=df)
plt.title('Correlation between Fed Rate and BTC price')
plt.xlabel('Fed Rate')
plt.ylabel('BTC Price')
plt.show()

# Convert the 'Date' column to a datetime object if it's not already
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# Create a line chart for Federal Funds Effective Rate
plt.figure(figsize=(10, 5), dpi=200)
plt.plot(df['Date'], df['DFF%'], label='Federal Interest Rate', color='red')
plt.title('Federal Funds Effective Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Federal Funds Effective Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert the 'Date' column to a datetime object if it's not already
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# Create a line chart for Bitcoin Price
plt.figure(figsize=(10, 5), dpi=200)
plt.plot(df['Date'], df['BTC_price'], label='Bitcoin Price over Time', color='red')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert the 'Date' column to a datetime object if it's not already
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# Create a line chart for Bitcoin Price
plt.figure(figsize=(10, 5), dpi=200)
plt.plot(df['Date'], df['Google_Trends'], label='Google Trends over Time', color='red')
plt.title('Google Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Google Trends')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a line chart for Bitcoin VS Fed
plt.figure(figsize=(10, 5), dpi=200)
plt.plot(df['DFF%'], df['BTC_price'], label='Bitcoin price', color='red')
plt.title('Bitcoin price VS Fed')
plt.xlabel('Fed Rate')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## Relation between Google Trends and BTC_price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Google_Trends', y='BTC_price', data=df)
plt.title('Correlation between Google Trends and BTC price')
plt.xlabel('Google Trends')
plt.ylabel('BTC Price')
plt.show()


## Relation between Mortgage Rates and BTC_Volume
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Mortgage_Rate_US', y='BTC_Volume', data=df)
plt.title('Correlation between Mortgage_Rate_US and BTC Volume')
plt.xlabel('Mortgage_Rate_US')
plt.ylabel('BTC Volume')
plt.show()

## Relation between Fear & Greed Index and BTC_price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Fear_Index', y='BTC_price', data=df)
plt.title('Correlation between Fear_Index and BTC price')
plt.xlabel('Fear Index')
plt.ylabel('BTC Price')
plt.show()


'''
# Create a lagged version of the Fear_Index column with a lag of 5 days
df['Fear_Index_Lagged'] = df['Fear_Index'].shift(10)
# Create a scatter plot
plt.figure(figsize=(10, 5), dpi=200)
plt.scatter(df['Fear_Index_Lagged'], df['BTC_price'], alpha=0.5)
plt.xlabel('Fear Index (Lagged)')
plt.ylabel('BTC Price')
plt.title('Scatter Plot: Fear Index vs. BTC Price (with 5-day Lag)')
plt.grid(True)
# Show the plot
plt.tight_layout()
plt.show()
'''



# Assuming your DataFrame is named 'df' and the 'Date' column is in datetime format.
# If not, you may need to convert the 'Date' column to datetime as shown earlier.
# Sort the DataFrame by 'Date' in ascending order to ensure consecutive days.
df = df.sort_values(by='Date')
# Calculate the differences between Fear_Index and BTC_price on consecutive days
df['Fear_Index_Diff'] = df['Fear_Index'].diff().shift(-1)
df['BTC_Price_Diff'] = df['BTC_price'].diff().shift(-1)
# Drop the last row as it won't have a valid next-day difference
df = df[:-1]
# Now, 'Fear_Index_Diff' contains the difference between day 1 and day 2, and
# 'BTC_Price_Diff' contains the difference between BTC prices on day 1 and day 2.
# You can now use these columns to analyze the relationship between them.

df

# Assuming you have already calculated 'Fear_Index_Diff' and 'BTC_Price_Diff' as mentioned earlier.

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['Fear_Index_Diff'], df['BTC_Price_Diff'], alpha=0.5)
plt.title("Fear Index Difference vs. BTC Price Difference")
plt.xlabel("Fear Index Difference (Day 1 to Day 2)")
plt.ylabel("BTC Price Difference (Day 1 to Day 2)")
plt.grid(True)
plt.show()

# Create a line chart for Bitcoin VS Fed
plt.figure(figsize=(10, 5), dpi=200)
plt.scatter(df['Nasdaq_close'], df['BTC_price'], label='Bitcoin price', color='red')
plt.title('Bitcoin price VS Nasdaq Close Value')
plt.xlabel('Nasdaq')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a line chart for Bitcoin VS Fed
plt.figure(figsize=(10, 5), dpi=200)
plt.scatter(df['Nasdaq_Volume'], df['BTC_price'], label='Bitcoin price', color='red')
plt.title('Bitcoin price VS Nasdaq Volume')
plt.xlabel('Nasdaq Volume')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create a line chart for Bitcoin VS Fed
plt.figure(figsize=(10, 5), dpi=200)
plt.scatter(df['DFF%'], df['Nasdaq_close'], label='Nasdaq vs Fed Rate', color='red')
plt.title('Fed Rate VS Nasdaq Close')
plt.xlabel('Fed Rate')
plt.ylabel('Nasdaq Close')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



X = df.drop(['BTC_price', 'Date'], axis=1)  # Features (independent variables)
y = df['BTC_price']  # Target variable (dependent variable)


# Create an imputer
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy like 'median' or 'most_frequent'
# Fit and transform your data to fill in missing values
X = imputer.fit_transform(X)


## splitting our data
## 80% for training 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

# standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

## create a dataframe for prediction
result_LR = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(result_LR)

# Calculate the absolute percentage difference
result_LR['Percentage_Difference'] = ((result_LR['Predicted'] - result_LR['Actual']) / result_LR['Actual']) * 100

# Print the DataFrame with the percentage difference column
print(result_LR)

# Calculate the average percentage difference
average_percentage_difference_LR = result_LR['Percentage_Difference'].mean()

# Print the average percentage difference
print(f'Average Percentage Difference for LR: {average_percentage_difference_LR:.2f}%')



##output_file_path = 'linear_regression_predictions.csv'
# Save the DataFrame to a CSV file
##prediction_Linear_Regression.to_csv(output_file_path, index=False)

## model evaluation
## Mean absolute error
## Mean square error
## Root mean square error
## Rsquared --0.75, 0.82, 0.91, 0.97, 0.99, 1

print('MAE', metrics.mean_absolute_error(y_test, y_pred))
print('MSE', metrics.mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2', metrics.r2_score(y_test, y_pred))


new_data_LR = pd.DataFrame({
    'Fear_Index': [74],
    'BTC_Volume': [12796779678],
    'ADA_price': [0.34],  
    'ADA_Volume': [317707347],
    'BNB_price': [244.33],
    'BNB_volume': [571460392],
    'ETH_price': [1895.94],
    'ETH_volume': [12950158460],
    'LINK_price': [12.23],
    'LINK_volume': [880121873],
    'XLM_price': [0.13],
    'XLM_volume': [111814111],
    'DFF%': [5.33],
    'Google_Trends': [49],
    'Mortgage_Rate_US': [7.76],
    'Nasdaq_close': [13478.28027],
    'Nasdaq_Volume': [4918750000],
    'Fear_Index_Diff': [0],
    'BTC_Price_Diff':[0],
})

# Use the scaler to transform the input data
scaled_new_data_LR = scaler.transform(new_data_LR)

# Make the prediction using the model
new_prediction_LR = lin_reg.predict(scaled_new_data_LR)

# Access the first element of the NumPy array and format it
formatted_prediction_LR = "{:.2f}".format(new_prediction_LR[0])

# Print the formatted prediction
print(f"The predicted Linear Regression Bitcoin price is: ${formatted_prediction_LR}")



# Build the ANN model
model = Sequential()
model.add(Dense(units=128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Data: {loss}')

# Plot training history 
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Make predictions
y_pred_ANN = model.predict(X_test)

# Visualize the results 
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='True Prices', alpha=0.7)
plt.plot(y_pred_ANN, label='Predicted Prices', alpha=0.7)
plt.xlabel('Samples')
plt.ylabel('BTC Price')
plt.legend()
plt.show()


## ANN


import pandas as pd
import numpy as np

# Assuming y_test and y_pred are not 1-dimensional, convert them to Series
y_test = pd.Series(y_test)

# Flatten y_pred using .ravel() or .flatten()
y_pred_ANN = y_pred_ANN.ravel()  # or y_pred.flatten()

# Create a DataFrame to compare actual and predicted values
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_ANN})

# Print the DataFrame
print(result_df)


# Specify the file path where you want to save the CSV file
file_path = "result_ANN.csv"

# Export the DataFrame to a CSV file
result_df.to_csv(file_path, index=False)  # Set index=False to exclude the index column

# Print a message to confirm that the export was successful
print(f"Result DataFrame has been exported to {file_path}")

# Calculate the absolute percentage difference
result_df['Percentage_Difference'] = ((result_df['Predicted'] - result_df['Actual']) / result_df['Actual']) * 100

# Print the DataFrame with the percentage difference column
print(result_df)

# Calculate the average percentage difference
average_percentage_difference = result_df['Percentage_Difference'].mean()

# Print the average percentage difference
print(f'Average Percentage Difference for ANN: {average_percentage_difference:.2f}%')


new_data = pd.DataFrame({
    'Fear_Index': [74],
    'BTC_Volume': [12796779678],
    'ADA_price': [0.34], 
    'ADA_Volume': [317707347],
    'BNB_price': [244.33],
    'BNB_volume': [571460392],
    'ETH_price': [1895.94],
    'ETH_volume': [12950158460],
    'LINK_price': [12.23],
    'LINK_volume': [880121873],
    'XLM_price': [0.13],
    'XLM_volume': [111814111],
    'DFF%': [5.33],
    'Google_Trends': [49],
    'Mortgage_Rate_US': [7.76],
    'Nasdaq_close': [13478.28027],
    'Nasdaq_Volume': [4918750000],
    'Fear_Index_Diff': [0],
    'BTC_Price_Diff':[0],
})

# Use the scaler to transform the input data
scaled_new_data = scaler.transform(new_data)

# Make the prediction using the model
new_prediction = model.predict(scaled_new_data)

# Format the predicted Bitcoin price to display with 2 decimal places
formatted_prediction = "{:.2f}".format(new_prediction[0][0])

# Print the formatted prediction
print(f"The predicted Bitcoin price is: ${formatted_prediction}")














import pandas as pd
import matplotlib.pyplot as plt

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Specify the lag days
lag_days = 365

# Create a new column for future Bitcoin price
df['Future_BTC_price'] = df['BTC_price'].shift(-lag_days)

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Google_Trends'], label='Google Trends', color='blue')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['Future_BTC_price'], label=f'Future Bitcoin Price ({lag_days} days)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Future Bitcoin Price', color='green')
ax2.set_ylabel('Google Trends', color='blue')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Specify the lag days
lag_days = 365

# Create a new column for future Bitcoin price
df['Future_BTC_price'] = df['BTC_price'].shift(-lag_days)

# Select relevant columns
selected_columns = ['Google_Trends', 'Future_BTC_price']
correlation_df = df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = correlation_df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title(f'Correlation Map: Google Trends vs. Future Bitcoin Price ({lag_days} days later)')
plt.show()

correlation_matrix

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['ADA_price'], label='ADA Price', color='blue')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('ADA Price', color='blue')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()


# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['BNB_price'], label='BNB Price', color='black')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('BNB Price', color='black')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['ETH_price'], label='ETH Price', color='purple')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('ETH Price', color='purple')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['LINK_price'], label='LINK Price', color='cyan')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('LINK Price', color='cyan')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['XLM_price'], label='XLM Price', color='yellow')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('XLM Price', color='yellow')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Fear_Index'], label='Fear & Greed Index', color='red')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('Fear & Greed Index', color='red')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

# Plot the data
plt.figure(figsize=(10, 5), dpi=200)

# Plot Google Trends with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=200)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['DFF%'], label='FED Rate', color='gray')

# Plot Future Bitcoin price
ax1.plot(df['Date'], df['BTC_price'], label=f'Bitcoin Price)', color='green')

# Set labels and legends
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price', color='green')
ax2.set_ylabel('Fed Rate', color='gray')

# Show legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Grid
ax1.grid(True)
ax2.grid(True)

# Tight layout
plt.tight_layout()
plt.show()

df