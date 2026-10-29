# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first (for better Docker caching)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
