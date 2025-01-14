import cv2
from PIL import Image
import io
import requests
import boto3
from google.cloud import storage
from azure.storage.blob import BlobServiceClient
import tweepy
import facebook

class ImageExporter:
    def __init__(self, twitter_credentials=None, facebook_credentials=None, aws_credentials=None, gcp_credentials=None, azure_credentials=None):
        self.twitter_api = self.init_twitter_api(twitter_credentials) if twitter_credentials else None
        self.facebook_api = self.init_facebook_api(facebook_credentials) if facebook_credentials else None
        self.s3_client = boto3.client('s3', **aws_credentials) if aws_credentials else None
        self.gcs_client = storage.Client.from_service_account_json(gcp_credentials) if gcp_credentials else None
        self.azure_blob_client = BlobServiceClient.from_connection_string(azure_credentials) if azure_credentials else None

    def export_image(self, image, format='jpg', quality=95):
        if format == 'jpg':
            return cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        elif format == 'png':
            return cv2.imencode('.png', image)[1]
        elif format == 'webp':
            return cv2.imencode('.webp', image, [int(cv2.IMWRITE_WEBP_QUALITY), quality])[1]
        elif format == 'tiff':
            return cv2.imencode('.tiff', image)[1]
        else:
            raise ValueError(f"Unsupported format: {format}")

    def share_to_social_media(self, image, platform, message=""):
        image_buffer = self.export_image(image, 'png')
        
        if platform.lower() == 'twitter':
            return self._share_to_twitter(image_buffer, message)
        elif platform.lower() == 'facebook':
            return self._share_to_facebook(image_buffer, message)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def upload_to_cloud(self, image, cloud_service, bucket_name, file_name):
        image_buffer = self.export_image(image, 'png')
        
        if cloud_service.lower() == 'aws':
            return self._upload_to_aws_s3(image_buffer, bucket_name, file_name)
        elif cloud_service.lower() == 'gcp':
            return self._upload_to_google_cloud_storage(image_buffer, bucket_name, file_name)
        elif cloud_service.lower() == 'azure':
            return self._upload_to_azure_blob_storage(image_buffer, bucket_name, file_name)
        else:
            raise ValueError(f"Unsupported cloud service: {cloud_service}")

    def init_twitter_api(self, credentials):
        auth = tweepy.OAuthHandler(credentials['consumer_key'], credentials['consumer_secret'])
        auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])
        return tweepy.API(auth)

    def init_facebook_api(self, credentials):
        return facebook.GraphAPI(access_token=credentials['access_token'])

    def _share_to_twitter(self, image_buffer, message):
        if not self.twitter_api:
            raise ValueError("Twitter API not initialized")
        try:
            media = self.twitter_api.media_upload("image.png", file=io.BytesIO(image_buffer))
            self.twitter_api.update_status(status=message, media_ids=[media.media_id])
            return True
        except Exception as e:
            print(f"Error sharing to Twitter: {str(e)}")
            return False

    def _share_to_facebook(self, image_buffer, message):
        if not self.facebook_api:
            raise ValueError("Facebook API not initialized")
        try:
            self.facebook_api.put_photo(image=io.BytesIO(image_buffer), message=message)
            return True
        except Exception as e:
            print(f"Error sharing to Facebook: {str(e)}")
            return False

    def _upload_to_aws_s3(self, image_buffer, bucket_name, file_name):
        if not self.s3_client:
            raise ValueError("AWS S3 client not initialized")
        try:
            self.s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=image_buffer)
            return True
        except Exception as e:
            print(f"Error uploading to AWS S3: {str(e)}")
            return False

    def _upload_to_google_cloud_storage(self, image_buffer, bucket_name, file_name):
        if not self.gcs_client:
            raise ValueError("Google Cloud Storage client not initialized")
        try:
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            blob.upload_from_string(image_buffer, content_type='image/png')
            return True
        except Exception as e:
            print(f"Error uploading to Google Cloud Storage: {str(e)}")
            return False

    def _upload_to_azure_blob_storage(self, image_buffer, container_name, blob_name):
        if not self.azure_blob_client:
            raise ValueError("Azure Blob Storage client not initialized")
        try:
            container_client = self.azure_blob_client.get_container_client(container_name)
            container_client.upload_blob(name=blob_name, data=image_buffer, overwrite=True)
            return True
        except Exception as e:
            print(f"Error uploading to Azure Blob Storage: {str(e)}")
            return False

# Usage example:
# twitter_creds = {
#     'consumer_key': 'your_consumer_key',
#     'consumer_secret': 'your_consumer_secret',
#     'access_token': 'your_access_token',
#     'access_token_secret': 'your_access_token_secret'
# }
# facebook_creds = {'access_token': 'your_facebook_access_token'}
# aws_creds = {'aws_access_key_id': 'your_access_key', 'aws_secret_access_key': 'your_secret_key'}
# gcp_creds = 'path/to/your/gcp_credentials.json'
# azure_creds = 'your_azure_connection_string'

# exporter = ImageExporter(twitter_credentials=twitter_creds, facebook_credentials=facebook_creds, 
#                          aws_credentials=aws_creds, gcp_credentials=gcp_creds, azure_credentials=azure_creds)
# image = cv2.imread('path/to/image.jpg')

# Export image
# jpg_buffer = exporter.export_image(image, 'jpg', 95)
# with open('exported_image.jpg', 'wb') as f:
#     f.write(jpg_buffer)

# Share to social media
# exporter.share_to_social_media(image, 'twitter', "Check out this image!")

# Upload to cloud
# exporter.upload_to_cloud(image, 'aws', 'my-bucket', 'image.png')
