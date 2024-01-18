PROJECT_DIR=$(cd ../../..; pwd)
sed -i "s@PROJECT_DIR@${PROJECT_DIR}@g" trackers.ini
