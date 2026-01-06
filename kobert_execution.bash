#!/bin/bash

echo " " | sudo -S chmod 600 /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key

ssh -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key ubuntu@146.56.111.104 "sudo chown ubuntu:ubuntu /home/ubuntu/*"

while true
do
    scp -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key ubuntu@146.56.111.104:/home/ubuntu/user/* /home/wego/KoBERT_ws/src/user/

    python3 /home/wego/KoBERT_ws/src/kobert_question.py
    
    scp -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key /home/wego/KoBERT_ws/src/question/* ubuntu@146.56.111.104:/home/ubuntu/question/

    scp -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key ubuntu@146.56.111.104:/home/ubuntu/answer/* /home/wego/KoBERT_ws/src/answer/


    python3 /home/wego/KoBERT_ws/src/kobert_result.py
    
    scp -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key /home/wego/KoBERT_ws/src/result/* ubuntu@146.56.111.104:/home/ubuntu/result/

    rm -rf /home/wego/KoBERT_ws/src/script/user/*
    rm -rf /home/wego/KoBERT_ws/src/script/question/*
    rm -rf /home/wego/KoBERT_ws/src/script/answer/*
    rm -rf /home/wego/KoBERT_ws/src/script/result/*

    sleep 10
done

    # scp -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key /home/wego/KoBERT_ws/src/script/user/ ubuntu@146.56.111.104:/home/ubuntu/user/
    # scp -i /home/wego/KoBERT_ws/src/ssh-key-2024-09-22.key /home/wego/KoBERT_ws/src/script/answer/ ubuntu@146.56.111.104:/home/ubuntu/answer/
