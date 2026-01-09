#!/bin/bash

KEY_PATH="path to key"              # 예: /home/admin/KoBERT_ws/src/ssh-key-2024-09-22.key
LOCAL_BASE="path to local"            # 예: /home/admin/KoBERT_ws/src
REMOTE_BASE="path to remote"           # 예: /home/ubuntu
REMOTE_USER="ubuntu"
REMOTE_HOST="146.56.111.104"

echo " " | sudo -S chmod 600 "$KEY_PATH"

ssh -i "$KEY_PATH" ${REMOTE_USER}@${REMOTE_HOST} "sudo chown ubuntu:ubuntu ${REMOTE_BASE}/*"

while true
do
    scp -i "$KEY_PATH" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/user/* "${LOCAL_BASE}/user/"

    python3 "${LOCAL_BASE}/kobert_question.py"
    
    scp -i "$KEY_PATH" "${LOCAL_BASE}/question/*" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/question/

    scp -i "$KEY_PATH" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/answer/* "${LOCAL_BASE}/answer/"


    python3 "${LOCAL_BASE}/kobert_result.py"
    
    scp -i "$KEY_PATH" "${LOCAL_BASE}/result/*" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/result/

    rm -rf "${LOCAL_BASE}/script/user/*"
    rm -rf "${LOCAL_BASE}/script/question/*"
    rm -rf "${LOCAL_BASE}/script/answer/*"
    rm -rf "${LOCAL_BASE}/script/result/*"

    sleep 10
done
