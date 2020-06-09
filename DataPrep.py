
%%time
import boto3
import re
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
import os 
import urllib.request
import boto3

#Download and Unzip
!wget https://yourDatasource
!unzip https://yourdatasource.zip


role = get_execution_role()
bucket = sagemaker.Session().default_bucket()
training_image = get_image_uri(boto3.Session().region_name, 'image-classification')

def upload_to_s3(channel, Path):
    s3_path_to_data = sagemaker.Session().upload_data(bucket=bucket, 
                                                  path=Path, 
                                                  key_prefix=channel)


# data copy to s3
s3_train_key = "image-classification-full-training/train"
s3_validation_key = "image-classification-full-training/validation"
s3_train = 's3://{}/{}/'.format(bucket, s3_train_key)
s3_validation = 's3://{}/{}/'.format(bucket, s3_validation_key)


upload_to_s3(s3_train_key, '/home/ec2-user/SageMaker/chest_xray/train/')

upload_to_s3(s3_validation_key, '/home/ec2-user/SageMaker/chest_xray/val/')



