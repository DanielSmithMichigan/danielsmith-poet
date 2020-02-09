# Goal:
# 1. Accept SQS message with:
#   A. Random seed
#   B. Model URL
# 2. Download Model
# 3. Perturb model weights using random seed
# 4. Report random seed back to parent

import boto3

client = boto3.client('sqs')

response = client.receive_message(
    QueueUrl='https://sqs.us-west-2.amazonaws.com/295909865373/seeds',
    MaxNumberOfMessages=1,
    VisibilityTimeout=max_task_time_sec + 10,
    WaitTimeSeconds=10
)

"""
{
    'Messages': [
        {
            'MessageId': 'string',
            'ReceiptHandle': 'string',
            'MD5OfBody': 'string',
            'Body': 'string',
            'Attributes': {
                'string': 'string'
            },
            'MD5OfMessageAttributes': 'string',
            'MessageAttributes': {
                'string': {
                    'StringValue': 'string',
                    'BinaryValue': b'bytes',
                    'StringListValues': [
                        'string',
                    ],
                    'BinaryListValues': [
                        b'bytes',
                    ],
                    'DataType': 'string'
                }
            }
        },
    ]
}
"""