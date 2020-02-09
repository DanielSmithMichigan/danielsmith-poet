# Goal:
# 1. Accept SQS message with:
#   A. Random seed
#   B. Model URL
# 2. Download Model
# 3. Perturb model weights using random seed
# 4. Report random seed back to parent
import shutil
import requests
from os import path
import boto3
s3 = boto3.client('s3')
sqs = boto3.resource('sqs')
import json

import tensorflow as tf
import numpy as np
import gym

from uuid import uuid4
from config import *

# token = requests.put('http://169.254.169.254/api/token').text
# headers = {'X-aws-ec2-metadata-token': token}
# local_ip = requests.get('http://169.254.169.254/latest/meta-data/local-ipv4', headers=headers).text

# Get the queue
queue = sqs.get_queue_by_name(QueueName=queue_name)
for i in range(7):
    response = queue.send_messages(Entries=[
        {
            'Id': str(uuid4()),
            'MessageBody': "Blank",
            'MessageAttributes': {
                'ModelSuffix': {
                    'StringValue': '33f4a45e-0345-449a-aa32-e36b42ea8e49',
                    'DataType': 'String'
                },
                'RandomSeed': {
                    'StringValue': str(np.random.randint(low=100000, high=999999)),
                    'DataType': 'String'
                }
            }
        }
    ])

print(response)