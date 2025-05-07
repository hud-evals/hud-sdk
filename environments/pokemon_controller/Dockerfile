# Use Python 3.8 as base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \                                                                                                                                
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

<<<<<<< HEAD
# Copy the package files 
=======
# Copy the package files
>>>>>>> 92f0d50 (working local docker)
RUN mkdir -p /controller
COPY src /controller/src
COPY pyproject.toml /controller/pyproject.toml
COPY gamefiles ./gamefiles


# Install the package in development mode
RUN pip install -e /controller --break-system-packages

# Set the command to run the controller
<<<<<<< HEAD
CMD ["tail", "-f", "/dev/null"]
=======
CMD ["python", "-m", "hud_controller.main"]
>>>>>>> 92f0d50 (working local docker)
