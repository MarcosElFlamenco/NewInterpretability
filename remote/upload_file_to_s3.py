import boto3
import os
import argparse

def upload_files_to_s3(bucket_name, file_paths, s3_folder=""):
    """
    Uploads a list of files to a specified S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_paths (list of str): List of file paths to upload.
        s3_folder (str, optional): Folder in S3 where files will be uploaded. Default is root.
    """
    s3_client = boto3.client('s3')
    
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue

        file_name = os.path.basename(file_path)
        s3_key = os.path.join(s3_folder, file_name) if s3_folder else file_name

        try:
            s3_client.upload_file(file_path, bucket_name, s3_key)
            print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a CSV file with transcripts.')
    parser.add_argument('--bucket_name', type=str, help='The path to the input CSV file')
    parser.add_argument('--file_paths', nargs='+', type=str, help='The length to truncate the transcript to')

    args = parser.parse_args()

    upload_files_to_s3(args.bucket_name, args.file_paths, s3_folder="")
