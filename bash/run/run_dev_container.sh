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
           -p "${APP_HOST_PORT}:${APP_SERVER_PORT}" \
           -p "8888:8888" \
           -v "./:/app/" \
           -v '/tmp/.X11-unix:/tmp/.X11-unix' \
           -e DISPLAY=$DISPLAY \
            --name  "${APP_CNTNR_NAME}_dev2" \
           "${DOCKER_IMG}" sh -c \
           "${command}"

# show all running dockers 
docker ps

# #### -----------------------   RUNNING THE PROJECT DOCKER-COMPOSE  -------------------------------
# echo && echo "[${PROJECT_NAME}][Docker-compose[Jupyter] running the development container(s)..."
# docker-compose  -p "${PROJECT_NAME}" -f docker-compose.yml up -d 
