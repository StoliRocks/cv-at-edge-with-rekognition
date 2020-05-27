import boto3
import json
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from botocore.exceptions import ClientError

def generate_presigned_url(photo, bucket, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket,
                                                            'Key': photo},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response

def lambda_handler(event, context):

    # logger.info(event)
    # logger.info(context)

    photo=event["DetectLabels"]['key']
    bucket=event["DetectLabels"]['bucketName']
    response=generate_presigned_url(photo, bucket)
    logger.info(response)
    return response
