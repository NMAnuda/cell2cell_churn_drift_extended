import boto3

def deploy_model(model_path, bucket='cell2cell-bucket'):
    """
    Deploy model to S3 + EC2 (mock).
    """
    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket, 'deployed_model.pkl')
    print(f"✅ Model deployed to s3://{bucket}/deployed_model.pkl")

    ec2 = boto3.client('ec2')
    response = ec2.run_instances(ImageId='ami-0abcdef1234567890', InstanceType='t3.micro', MinCount=1)
    instance_id = response['Instances'][0]['InstanceId']
    print(f"✅ EC2 deployed: {instance_id}")

# Usage
deploy_model('models/retrained_model.pkl')