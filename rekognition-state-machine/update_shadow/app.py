import boto3
import json
import time
import uuid

client = boto3.client('iot-data', region_name='us-east-1')
thingName = 'Garage1'
shadowTopic = '$aws/things/Garage1/shadow/update'
logRecordTopic = 'garage/001/bay/02/logs'

def publish_to_topic(topic,data):

    response = client.publish(
        topic=topic,
        payload=data
    )

    return response

def get_thing_shadow(thingName):

    response = client.get_thing_shadow(
        thingName=thingName
    )

    results_iterator = response['payload'].iter_lines()
    results = [r for r in results_iterator]
    results = results[0].decode()
    results = json.loads(results)

    return results

def lambda_handler(event, context):

    # newStartTime = event['payload']['state']['variables']['time']
    newStartTime = int(round(time.time() * 1000))
    previousStartTime = None
    lastActualMs = 0
    actualMs = 0
    newState = 'Vacant'
    previousState = 'Occupied'

    shadow = get_thing_shadow(thingName)

    try:
        previousStartTime = shadow['state']['reported']['start_time']
    except:
        print('No start_time in shadow.')

    try:
        lastActualMs = shadow['state']['reported']['last_actual_ms']
    except:
        print('No last_actual_ms in shadow.')

    try:
        actualMs = shadow['state']['reported']['actual_ms']
    except:
        print('No actual_ms in shadow.')

    if event["eventName"] == "PublishOccupiedShadow":
        newState = 'Occupied'
        previousState = 'Vacant'

    payload = { 
        "state": { 
            "reported": {
                "vacancy": newState,
                "start_time": newStartTime,
                "last_actual_ms": actualMs
            } 
        }
    }

    publish_to_topic(shadowTopic, json.dumps(payload))

    logRecord = {
        "id": str(uuid.uuid4()),
        "start_ms": previousStartTime,
        "end_ms": newStartTime,
        "duration_ms": newStartTime-previousStartTime,
        "state": previousState,
        "duration_actual_ms": actualMs - lastActualMs,
        "url": event['payload']['state']['variables']['url']
    }

    publish_to_topic(logRecordTopic, json.dumps(logRecord))

    return {
        "statusCode": 200
    }
