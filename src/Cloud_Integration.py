from google.cloud import storage
from azure.storage.blob import BlobServiceClient

class CloudManager:
    def __init__(self):
        self.gcs_client = storage.Client()
        self.azure_client = BlobServiceClient.from_connection_string("your_connection_string")

    def upload_to_gcs(self, bucket_name, source_file_name, destination_blob_name):
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

    def download_from_gcs(self, bucket_name, source_blob_name, destination_file_name):
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

    def upload_to_azure(self, container_name, source_file_name, destination_blob_name):
        container_client = self.azure_client.get_container_client(container_name)
        with open(source_file_name, "rb") as data:
            container_client.upload_blob(name=destination_blob_name, data=data)

    def download_from_azure(self, container_name, source_blob_name, destination_file_name):
        container_client = self.azure_client.get_container_client(container_name)
        with open(destination_file_name, "wb") as file:
            blob_data = container_client.download_blob(source_blob_name)
            blob_data.readinto(file)
