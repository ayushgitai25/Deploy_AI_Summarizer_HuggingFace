# Use Python 3.11 slim image as the base image
# Slim version reduces image size while including essential Python components
FROM python:3.11-slim

# Set the working directory inside the container to /app
# All subsequent commands will be executed from this directory
WORKDIR /app

# Update package list and install system dependencies
# build-essential: Compilers needed for some Python packages
# curl: Required for health check functionality
# Clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt file first for better Docker layer caching
# This allows dependency installation to be cached separately from app code
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
# --no-cache-dir prevents pip from storing cache files (reduces image size)
# -r flag tells pip to install from requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining application files into the container
# This happens after dependency installation for better caching
COPY . ./

# Set Streamlit configuration environment variables
# STREAMLIT_HOME: Prevents permission issues by using /tmp directory
ENV STREAMLIT_HOME=/tmp/.streamlit

# Disable Streamlit usage statistics collection
# This prevents analytics data from being sent to Streamlit servers
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Document that the container will listen on port 8501
# This is Streamlit's default port and required for Hugging Face Spaces
EXPOSE 8501

# Add health check to verify the application is running properly
# Curl command checks Streamlit's internal health endpoint
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Define the command that runs when container starts
# streamlit run: Starts the Streamlit application
# app.py: The main application file
# --server.port=8501: Sets the port (required for Hugging Face Spaces)
# --server.address=0.0.0.0: Makes app accessible from outside container
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
