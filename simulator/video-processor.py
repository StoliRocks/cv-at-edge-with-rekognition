import json
import cv2
import numpy as np
import boto3
import time

bucketName = 'loves-rekognition-imagebucket-d08f31xbuk00'
keyPrefix = 'loves-video-frame-'
# Playing video from file:
cap = cv2.VideoCapture('local.mp4')
s3 = boto3.resource('s3')


def update_shadow(payload):

    client = boto3.client('iot-data', region_name='us-east-2')

    data = json.dumps(payload)

    response = client.publish(
        topic='$aws/things/rek_at_edge_Core/shadow/update',
        payload=data
    )

    return response


start = int(round(time.time() * 1000))

payload = {
    "state": {
        "reported": {
            "start_time": start,
            "last_actual_ms": 0
        }
    }
}
update_shadow(payload)

currentFrame = 0
while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    if currentFrame % 50 == 0:

        # Saves image of the current frame in jpg file
        name = keyPrefix + str(currentFrame) + '.jpg'

        millis = cap.get(cv2.CAP_PROP_POS_MSEC)

        millis = int(millis)
        seconds = (millis/1000) % 60
        seconds = int(seconds)
        minutes = (millis/(1000*60)) % 60
        minutes = int(minutes)
        hours = (millis/(1000*60*60)) % 24

        print('Creating...' + name + ' - ' + "%02d:%02d:%02d" %
              (hours, minutes, seconds))

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame = cv2.putText(frame, "%02d:%02d:%02d" % (hours, minutes, seconds), (
            25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        image_string = cv2.imencode('.jpg', frame)[1].tostring()

        s3Object = s3.Object(bucketName, name)
        s3Object.put(Body=image_string)

        payload = {
            "state": {
                "reported": {
                    "actual_ms": millis
                }
            }
        }
        update_shadow(payload)

        time.sleep(3)

    currentFrame += 1
print('Total frames: ' + str(currentFrame))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
