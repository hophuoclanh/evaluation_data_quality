# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt (if you have one)
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run your module when the container launches (adjust as needed)
# Here, just as an example, we're assuming you want to run your module as a script.
# If you have a different entry point or if you're running a web server, adjust accordingly.
CMD ["python", "data_quality_metrics.py"]
