# Use an official Python runtime as a parent image
FROM python:3.12 as pv

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --root-user-action=ignore --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 80

# Run Streamlit app and FASTAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


