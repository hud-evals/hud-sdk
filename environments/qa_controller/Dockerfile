FROM python:3.10-alpine

# Install build dependencies
RUN apk add --no-cache build-base

# Create directories
RUN mkdir -p /tmp/state /environment

# Set working directory
WORKDIR /environment

# The volume mount will handle copying the files
# pip install -e . will happen during initialization

# Keep container running
CMD ["tail", "-f", "/dev/null"]
