import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load data
df_original = pd.read_csv('merged_data.csv')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Collect user input features
user_input = {
    'Fear_Index': st.sidebar.number_input('Fear Index', value=74),
    'BTC_Volume': st.sidebar.number_input('BTC Volume', value=12796779678),
    'ADA_price': st.sidebar.number_input('ADA Price', value=0.34),
    'ADA_Volume': st.sidebar.number_input('ADAVolume', value=317707347),
    'BNB_price': st.sidebar.number_input('BNB Price', value=244.33),
    'BNB_volume': st.sidebar.number_input('BNB Volume', value=571460392),
    'ETH_price': st.sidebar.number_input('ETH Price', value=1895.94),
    'ETH_volume': st.sidebar.number_input('ETH Volume', value=12950158460),
    'LINK_price': st.sidebar.number_input('LINK Price', value=12.23),
    'LINK_volume': st.sidebar.number_input('LINK Volume', value=80121873),
    'XLM_price': st.sidebar.number_input('XLM Price', value=0.13),
    'XLM_volume': st.sidebar.number_input('XLM Volume', value=111814111),
    'DFF%': st.sidebar.number_input('FED Interest Rate', value=5.33),
    'Google_Trends': st.sidebar.number_input('Google Trends Score', value=49),
    'Mortgage_Rate_US': st.sidebar.number_input('Average Mortgage Rate in US', value=7.76),
    'Nasdaq_close': st.sidebar.number_input('Nasdaq Close Value', value=13478.28027),
    'Nasdaq_Volume': st.sidebar.number_input('Nasdaq Volume', value=4918750000),
   }

# Display user input
st.write('## User Input Features')
st.write(pd.DataFrame([user_input]))

# Model selection
model_option = st.sidebar.radio('Select Model', ['Linear Regression', 'Artificial Neural Network'])

# Preprocess user input for prediction
new_data = pd.DataFrame([user_input])

# Standardize the new data using the same scaler as in your original code
scaler = StandardScaler()

# Assuming you have some columns to drop and convert to numeric
X = df_original.drop(['BTC_price', 'Date'], axis=1)
X = pd.get_dummies(X, drop_first=True)

# Convert any remaining non-numeric columns to numeric (if needed)
X = X.apply(pd.to_numeric, errors='coerce')

# Use SimpleImputer to handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data
y = df_original['BTC_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

# Standardize the numeric features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

new_prediction = None  # or any default value you prefer

# Make predictions based on the selected model
if model_option == 'Linear Regression':
    # Create and fit the linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    new_prediction = lin_reg.predict(scaler.transform(new_data))

else:
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

    # Make predictions
    new_prediction = model.predict(scaler.transform(new_data))
    print(f'Dimensions of new_prediction: {new_prediction.shape}')

    # ... (Visualization code for the ANN model)

# Display prediction
st.write('## Prediction')
formatted_prediction = "{:.2f}".format(float(new_prediction[0])) if new_prediction.ndim > 1 else "{:.2f}".format(float(new_prediction))
st.write(f'The predicted Bitcoin price is: ${formatted_prediction}')


# ... (Data exploration and visualization code)

# Save the result to a CSV file
# Save the result to a CSV file
# Save the result to a CSV file
# Save the result to a CSV file
# Save the result to a CSV file
# Save the result to a CSV file
# Save the result to a CSV file
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': np.squeeze(new_prediction)})
result_df['Percentage_Difference'] = ((result_df['Predicted'] - result_df['Actual']) / result_df['Actual']) * 100
result_df.to_csv("result_ANN.csv", index=False)



# Calculate the average percentage difference
average_percentage_difference = result_df['Percentage_Difference'].mean()

# Print the average percentage difference
print(f'Average Percentage Difference for ANN: {average_percentage_difference:.2f}%')


