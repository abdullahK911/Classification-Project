# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /streamlit

# Copy the current directory contents into the container at /app
COPY . /streamlit

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit app and FASTAPI
CMD ["streamlit", "run", "streamlit.py"]


