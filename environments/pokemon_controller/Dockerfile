# Use Python 3.8 as base image
FROM socpite/pokemon_env:latest

# Copy the controller files 
COPY . /controller

# Install the package in development mode
RUN pip install -e /controller --break-system-packages

# Set the command to run the controller
CMD ["tail", "-f", "/dev/null"]
