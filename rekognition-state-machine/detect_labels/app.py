import boto3
import json
import logging
import time
import dateutil.parser
from datetime import datetime
logger = logging.getLogger()
logger.setLevel(logging.INFO)

imgWidth = 960
imgHeight = 540

bayJSON = {
    "BoundingBox": {
        "Width": 0.75,
        "Height": 0.90,
        "Left": 0.10,
        "Top": 0.10
    }
}

left = imgWidth * bayJSON['BoundingBox']['Left']
top = imgHeight * bayJSON['BoundingBox']['Top']
width = imgWidth * bayJSON['BoundingBox']['Width']
height = imgHeight * bayJSON['BoundingBox']['Height']

bayBox = left, top, width, height

def detect_labels(photo, bucket):

    client=boto3.client('rekognition')

    response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        MaxLabels=10)

    # print('Detected labels for ' + photo) 
    # print()   
    # for label in response['Labels']:
    #     print ("Label: " + label['Name'])
    #     print ("Confidence: " + str(label['Confidence']))
    #     print ("Instances:")
    #     for instance in label['Instances']:
    #         print ("  Bounding box")
    #         print ("    Top: " + str(instance['BoundingBox']['Top']))
    #         print ("    Left: " + str(instance['BoundingBox']['Left']))
    #         print ("    Width: " +  str(instance['BoundingBox']['Width']))
    #         print ("    Height: " +  str(instance['BoundingBox']['Height']))
    #         print ("  Confidence: " + str(instance['Confidence']))
    #         print()

        # print ("Parents:")
        # for parent in label['Parents']:
        #     print ("   " + parent['Name'])
        # print ("----------")
        # print ()
    return response['Labels']

def lambda_handler(event, context):

    logger.info(json.dumps(event['detail']['requestParameters']))
    logger.info(context)

    photo=event['detail']['requestParameters']['key']
    bucket=event['detail']['requestParameters']['bucketName']
    labels=detect_labels(photo, bucket)
    iso_datetime = dateutil.parser.isoparse(event['time'])
    print("Labels detected: " + str(len(labels)))
    filtered_labels = filter(lambda x : x['Name'] == 'Truck', labels)

    result = {}
    result["DetectLabels"] = {}
    # result["DetectLabels"]["time"] = int(round(datetime.timestamp(iso_datetime))*1000)
    result["DetectLabels"]["time"] = int(round(time.time() * 1000))
    result["DetectLabels"]["labels"] = list(filtered_labels)
    result["DetectLabels"]["bucketName"] = bucket
    result["DetectLabels"]["key"] = photo
    result["DetectLabels"]["iou"] = 0.0
    result["DetectLabels"]["confidence"] = 0.0

    if len(result["DetectLabels"]["labels"]) > 0:

        result["DetectLabels"]["confidence"] = result["DetectLabels"]["labels"][0]["Confidence"]

        detectionLeft = imgWidth * result["DetectLabels"]["labels"][0]["Instances"][0]['BoundingBox']['Left']
        detectionTop = imgHeight * result["DetectLabels"]["labels"][0]["Instances"][0]['BoundingBox']['Top']
        detectionWidth = imgWidth * result["DetectLabels"]["labels"][0]["Instances"][0]['BoundingBox']['Width']
        detectioHeight = imgHeight * result["DetectLabels"]["labels"][0]["Instances"][0]['BoundingBox']['Height']

        detectionBox = detectionLeft, detectionTop, detectionWidth, detectioHeight

        xA = max(bayBox[0], detectionBox[0])
        yA = max(bayBox[1], detectionBox[1])
        xB = min(bayBox[2], detectionBox[2])
        yB = min(bayBox[3], detectionBox[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxBayArea = (bayBox[2] - bayBox[0] + 1) * (bayBox[3] - bayBox[1] + 1)
        boxDetectionArea = (detectionBox[2] - detectionBox[0] + 1) * (detectionBox[3] - detectionBox[1] + 1)

        iou = interArea / float(boxBayArea + boxDetectionArea - interArea)

        result["DetectLabels"]["iou"] = iou

    return result
