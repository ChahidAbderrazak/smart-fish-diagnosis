#/bin/bash
#### -----------------------   PREPARING THE WORKSPACE  -------------------------------
docker system prune -f
clear
. .env

#### -----------------------   RUNNING THE PROJECT DOCKER  -------------------------------
# run the the project container(s)
echo && echo "[${PROJECT_NAME}][dev] running the development container(s)..."
docker-compose -p "${PROJECT_NAME}" -f docker-compose.yml up -d


#### ----------------   SHOW/UPDATE THE URLs/IP-ADRESSES -------------------------
sleep 10
bash bash/4-open-app-servres-in-browser.sh
