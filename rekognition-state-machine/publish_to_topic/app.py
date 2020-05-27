import boto3
import json

def publish_to_topic(data):

    client = boto3.client('iot-data')

    response = client.publish(
        topic='garage/001/bay/02',
        payload=data
    )

    return response

def lambda_handler(event, context):

    data=json.dumps(event)
    response=publish_to_topic(data)

    return {
        "statusCode": 200
    }
