# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY project.py .

# Copy data
COPY data /app/data

# Expose the port that Gradio will run on
EXPOSE 7860

# Command to run the application
CMD ["python", "project.py"]

