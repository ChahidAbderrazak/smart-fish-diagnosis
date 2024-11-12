#/bin/bash
. .env
command="bash "
# command="python src/webapp.py "
# command="python src/main.py "
# command="jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' "


# get the name of the project docker image 
DOCKER_IMG="datascience:v1"
docker system prune -f

#### -----------------------   RUNNING THE PROJECT DOCKER  -------------------------------
# run the the project container(s)
echo && echo "[${PROJECT_NAME}][Docker][dev] running the development container(s)..."
docker run -it --rm \
                --network host \
                -e DISPLAY=$DISPLAY \
                -v /tmp/.X11-unix/:/tmp/.X11-unix \
                -v $PWD:/app \
                -p "${APP_HOST_PORT}:${APP_SERVER_PORT}" \
                -p "8888:8888" \
                --name  "${APP_CNTNR_NAME}_dev" \
                "${DOCKER_IMG}" sh -c \
                "${command}"

                # --runtime nvidia \
                # -v "/home/jetson/Desktop/nvidia-inference:/app/nvidia-inference" \



# show all running dockers 
docker ps