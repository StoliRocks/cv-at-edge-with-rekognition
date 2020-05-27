# aws-rekognition-snapshot-serverless

## Rekognition SAM application

![Rekognition Snapshots](/docs/rekognition-stepfunction.png)

```
cd rekognition-state-machine
aws s3 mb s3://<YOUR_BUCKET_NAME>
sam package --template-file template.yaml --s3-bucket <YOUR_BUCKET_NAME> --output-template-file packaged.yaml
sam deploy --template-file ./packaged.yaml --stack-name <YOUR_STACK_NAME> --capabilities CAPABILITY_IAM
```   
