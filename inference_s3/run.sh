#!/bin/bash

# Step 1: Build the Docker image
echo "Building the Docker image..."
sudo docker build -t my_image .


# Step 2: Run the Docker container with input arguments
echo "Running the Docker container with input arguments..."
sudo docker run --shm-size=1g -v ~/.aws:/root/.aws my_image --city_ls 'royal-oak-michigan-2021' 'birmingham-michigan-2021' --percent_ls '100' '100' --month_ls 'mar' 'mar'