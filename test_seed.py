# Goal:
# 1. Accept SQS message with:
#   A. Random seed
#   B. Model URL
# 2. Download Model
# 3. Perturb model weights using random seed
# 4. Report random seed back to parent


import shutil
from os import path, mkdir, listdir
from os.path import join, isfile
import sys
import boto3
s3 = boto3.client('s3')
sqs = boto3.resource('sqs')
sqs_client = boto3.client('sqs')
import json

import tensorflow as tf
import numpy as np
import gym

import matplotlib
import matplotlib.pyplot as plt
plt.ion()

overview_figure = plt.figure()
rewards_graph = overview_figure.add_subplot(1, 1, 1)
rewards_over_time = []

def update_graph():
    rewards_graph.cla()
    rewards_graph.plot(rewards_over_time, label="Reward")
    overview_figure.canvas.draw()
    plt.pause(0.00001)

from uuid import uuid4

from config import *

queue = sqs.get_queue_by_name(QueueName=queue_name)

actions_ph = tf.keras.Input(
    shape=(num_observations,),
    name="actions_ph"
)

layers = [
    actions_ph
]
for i in layer_definition:
    layers.append(
        tf.keras.layers.Dense(i, activation='relu')
    )
layers.append(
    tf.keras.layers.Dense(num_actions)
)
model = tf.keras.Sequential(layers)

while True:
    response = []
    while len(response) == 0:
        print("Polling")
        response = queue.receive_messages(
            MaxNumberOfMessages=1,
            VisibilityTimeout=max_task_time_sec + 10,
            WaitTimeSeconds=10,
            MessageAttributeNames=['*']
            # AttributeNames=['*']
        )
    print("Test beginning")

    message = response[0]
    receipt_handle = message.receipt_handle
    queue_url = message.queue_url
    model_suffix = message.message_attributes.get('ModelSuffix').get('StringValue')
    random_seed = int(message.message_attributes.get('RandomSeed').get('StringValue'))

    current_model_identifier = ''
    if path.exists(model_namefile):
        current_model_identifier_handle = open(model_namefile)
        current_model_identifier = current_model_identifier_handle.read()
        current_model_identifier_handle.close()

    if current_model_identifier != model_suffix:
        print("Downloading model")
        current_model_identifier_handle = open(model_namefile, "w")
        current_model_identifier_handle.write(model_suffix)
        current_model_identifier_handle.close()
        if path.exists(model_dir):
            shutil.rmtree(model_dir)
        mkdir(model_dir)
        for obj in s3.list_objects(Bucket=bucket_name, Prefix=model_suffix)['Contents']:
            if obj["Key"] != model_suffix + "/":
                s3.download_file(bucket_name, obj['Key'], model_dir + '/' + obj['Key'].split('/')[-1])

    env = gym.make(env_name)

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model.load_weights(latest_checkpoint)

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    print("Perturbing weights")

    for i in range(len(model.layers)):
        new_weights = model.layers[i].get_weights()
        for weight_set_idx in range(len(new_weights)):
            weight_set = new_weights[weight_set_idx]
            for weight_idx in range(len(weight_set)):
                weight_set[weight_idx] = weight_set[weight_idx] + np.random.uniform() * weight_std
        model.layers[i].set_weights(new_weights)

    state = env.reset()
    done = False
    total_reward = 0
    print("Testing new model")
    for i in range(max_environment_steps):
        actions_chosen = model(
            np.reshape(state, [-1, num_observations])
        )
        nextState, reward, done, info = env.step(
            actions_chosen[0]
        )
        total_reward += reward
        state = nextState
        if done:
            break
    rewards_over_time.append(total_reward)
    # update_graph()
    dynamodb = boto3.client("dynamodb")
    updated_item = dynamodb.update_item(
        TableName='danielsmith-poet-iterations',
        Key={
            'id': {
                'S': model_suffix
            }
        },
        UpdateExpression='SET results = list_append(if_not_exists(results, :empty_list), :r) ADD iterations :c',
        ExpressionAttributeValues={
            ':c': {
                'N': '1'
            },
            ':r': {
                'L': [
                    {
                        'L': [{
                            'S': str(random_seed)
                        }, {
                            'S': str(total_reward)
                        }]
                    }
                ]
            },
            ':empty_list': {
                'L': []
            }
        },
        ReturnValues='UPDATED_NEW'
    )

    if int(updated_item["Attributes"]["iterations"]["N"]) >= batch_size:
        try:
            updated_item = dynamodb.update_item(
                TableName='danielsmith-poet-iterations',
                Key={
                    'id': {
                        'S': model_suffix
                    }
                },
                UpdateExpression='SET next_step_initiated = :b',
                ConditionExpression='attribute_not_exists(next_step_initiated)',
                ExpressionAttributeValues={
                    ':b': {
                        'BOOL': True
                    }
                },
                ReturnValues='ALL_NEW'
            )
            model.load_weights(latest_checkpoint)
            rewards = [float(entry["L"][1]["S"]) for entry in updated_item["Attributes"]["results"]["L"]]
            seeds = [int(entry["L"][0]["S"]) for entry in updated_item["Attributes"]["results"]["L"]]
            std = np.std(rewards)
            mean = np.mean(rewards)
            z_scores = [(reward - mean) / std for reward in rewards]
            for seed, z_score in zip(seeds, z_scores):
                tf.random.set_seed(seed)
                np.random.seed(seed)
                for i in range(len(model.layers)):
                    new_weights = model.layers[i].get_weights()
                    for weight_set_idx in range(len(new_weights)):
                        weight_set = new_weights[weight_set_idx]
                        for weight_idx in range(len(weight_set)):
                            weight_set[weight_idx] = weight_set[weight_idx] + learning_rate / (batch_size * weight_std) * np.random.uniform() * z_score
                    model.layers[i].set_weights(new_weights)

            new_model_suffix = str(uuid4())
            print("New model suffix " + new_model_suffix)
            current_model_identifier_handle = open(model_namefile, "w")
            current_model_identifier_handle.write(new_model_suffix)
            current_model_identifier_handle.close()
            shutil.rmtree(model_dir)
            mkdir(model_dir)
            model.save_weights(join(model_dir, new_model_suffix))
            for file_suffix in listdir(model_dir):
                file_path = join(model_dir, file_suffix)
                if isfile(file_path):
                    s3.upload_file(file_path, bucket_name, join(new_model_suffix, file_suffix))
            messages_remaining = batch_size
            while messages_remaining > 0:
                current_batch_entries = []
                for i in range(min(1, messages_remaining)):
                    current_batch_entries.append({
                        'Id': str(uuid4()),
                        'MessageBody': "Blank",
                        'MessageAttributes': {
                            'ModelSuffix': {
                                'StringValue': new_model_suffix,
                                'DataType': 'String'
                            },
                            'RandomSeed': {
                                'StringValue': str(np.random.randint(low=100000, high=999999)),
                                'DataType': 'String'
                            }
                        }
                    })
                queue.send_messages(Entries=current_batch_entries)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print("Exception")

    print("Deleting message")
    sqs_client.delete_message(
        QueueUrl = queue_url,
        ReceiptHandle = receipt_handle
    )

