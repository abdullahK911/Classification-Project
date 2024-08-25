# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /fastapi

# Copy the current directory contents into the container at /app
COPY requirements.txt requirements.txt

COPY . /model.pth /fastapi

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8000

# Run Streamlit app and FASTAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


