import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations

from joblib import load  # For saving the pipeline

from flask import Flask
from flask import request
from flask import jsonify

from waitress import serve

from custom_transformers import InteractionTransformer
from custom_transformers import CustomImputer

confidence_level = 0.95

app = Flask('house_price_prediction')

# Load the model outside the route function so it's only loaded once
model = load('XGB_pipeline.joblib')

def calculate_rmse(actual_prices, predicted_prices):
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    return rmse

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json(force=True)  # Get the JSON data sent to the endpoint
    input_data = pd.DataFrame([input_json])  # Convert the JSON data to a pandas DataFrame

    # Make predictions using the model (logged predictions)
    logged_predictions = model.predict(input_data)
    # Convert logged predictions back to original scale
    predictions = np.expm1(logged_predictions)
    # Return predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9696)