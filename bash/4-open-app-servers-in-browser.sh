#/bin/bash
#### -----------------------   APP SERVERS  -------------------------------
. .env

#### -----------------------   IP Adresses  -------------------------------
echo && echo "[${PROJECT_NAME}][IP Adresses] Getting the IP adresses of the different servers..."
# ##! for Dockerfile
# docker inspect -f '{{.Name}} - {{.NetworkSettings.IPv4Address }}' $(docker ps -aq) > .env-ip

##! for docker-compsoe
docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker ps -aq) > .env-ip

sed -i 's/ - /=/g' .env-ip
sed -i 's|/||g' .env-ip
# sed -i '/=$/d' .env-ip
# cat .env-ip
. .env-ip

#### -----------------------   OPEN RUNNING SERVERS  -------------------------------
echo "[${PROJECT_NAME}][Servers] Open the different servers on the browser"
sleep 10

#### -----------------------   DATABASE CPNTAINER  --------------------------------
echo && echo "MySQL_CNTNR_NAME=${MySQL_CNTNR_NAME}"
eval "MySQL_CNTNR_IP=\$$MySQL_CNTNR_NAME"
if [ "$MySQL_CNTNR_IP" != "" ] ; then
	DATABASE_URL="mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MySQL_CNTNR_IP}/${MYSQL_DATABASE}"
	echo "MySQL_CNTNR_IP=${MySQL_CNTNR_IP}" >>.env-ip
	echo "DATABASE_URL=${DATABASE_URL}" >>.env-ip
	echo && echo "MySQL_CNTNR_IP=${MySQL_CNTNR_IP}"
	echo "DATABASE_URL=${DATABASE_URL}"
fi

#### -----------------------   APP CPNTAINER  --------------------------------
echo && echo "APP_CNTNR_IP=${APP_CNTNR_IP}"
eval "APP_CNTNR_IP=\$$APP_CNTNR_NAME"
if [ "$APP_CNTNR_IP" != "" ] ; then
	APP_SERVER_URL="http://$APP_CNTNR_IP:${APP_HOST_PORT}"
else
	APP_SERVER_URL="http://localhost:${APP_HOST_PORT}"
fi
echo && echo "-- app server URL = ${APP_SERVER_URL}"
sed -i '/APP_SERVER_URL/d' .env-ip
echo "APP_SERVER_URL=${APP_SERVER_URL}" >>.env-ip


#### -----------------------   WEBAPP CPNTAINER  --------------------------------
eval "WEBAPP_CNTNR_IP=\$$WEBAPP_CNTNR_NAME"
if [ "$WEBAPP_CNTNR_IP" != "" ] ; then
	WEBAPP_SERVER_URL="http://$WEBAPP_CNTNR_IP:${WEBAPP_HOST_PORT}"
else
	WEBAPP_SERVER_URL="http://localhost:${WEBAPP_HOST_PORT}"
fi

echo && echo "-- WebAapp server URL = ${WEBAPP_SERVER_URL}"
sed -i '/WEBAPP_SERVER_URL/d' .env-ip
echo "WEBAPP_SERVER_URL=${WEBAPP_SERVER_URL}" >>.env-ip
# xdg-open "${WEBAPP_SERVER_URL}"

