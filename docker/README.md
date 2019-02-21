# Docker
Contains docker files for running code. 

## Dependencies
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## How to run
* Build
    ```
    sudo docker build -f <PATH_TO_DOCKERFILE> -t <IMAGE_NAME> <PATH_TO_STORE_IMAGE>
    ```
* Run
    ```
    sudo docker run --runtime=nvidia -v /path/to/folder/on/machine:/path/in/docker -it <IMAGE_NAME>
    ```