#/bin/bash
. .env
# get the name of the project docker image 
DOCKER_IMG="${IMG_BUILDER}:${VERSION}"

echo && echo "[${PROJECT_NAME}][Docker-Compose] Stopping all container(s)..."
docker-compose -p "${PROJECT_NAME}" -f docker-compose.yml stop

docker stop $(docker stop $(docker ps -a -q --filter ancestor=${DOCKER_IMG} --format="{{.ID}}"))
docker system prune -f