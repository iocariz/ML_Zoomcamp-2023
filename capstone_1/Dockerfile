# It's a good practice to pin a specific Python version
FROM python:3.11-slim

# Install pipenv
RUN pip install pipenv

# Set the working directory in the container
WORKDIR /app

# Copy Pipenv files
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --system --deploy

# Copy the rest of your application files
COPY predict.py custom_transformers.py XGB_pipeline.joblib ./

# Make port 9696 available to the world outside this container
EXPOSE 9696

# Define the command to run your application
# Remember to replace predict.py with the correct file if needed
CMD ["python", "predict.py"]
