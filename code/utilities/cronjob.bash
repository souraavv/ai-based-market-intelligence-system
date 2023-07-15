#!/bin/bash
source /home/baadalvm/anaconda3/bin/activate;
conda activate demo; 
python3 /home/baadalvm/Playground_Ronak_Sourav/Practice/crawlDaily.py >> /home/baadalvm/Playground_Ronak_Sourav/Practice/croncrawldaily.log 2>&1