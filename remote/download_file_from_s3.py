import boto3
import os
import argparse
from tqdm import tqdm

def download_file_from_s3(bucket_name, s3_key, download_path):
    """
    Downloads a file from an S3 bucket to a specified local path with a progress bar.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 object key (path to file in S3).
        download_path (str): Local path where the file will be saved.
    """
    s3_client = boto3.client('s3')
    print(download_path)
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    try:
        # Get the file size for progress bar
        file_info = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = file_info['ContentLength']

        with open(download_path, 'wb') as f, tqdm(
            total=file_size, unit='B', unit_scale=True, desc=s3_key, ncols=80
        ) as pbar:
            s3_client.download_fileobj(
                Bucket=bucket_name,
                Key=s3_key,
                Fileobj=f,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        print(f"\nDownloaded s3://{bucket_name}/{s3_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading s3://{bucket_name}/{s3_key}: {e}")

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download multiple files from S3 with progress tracking."
    )
    parser.add_argument(
        '--bucket_name',
        type=str,
        required=True,
        help='Name of the S3 bucket.'
    )
    parser.add_argument(
        '--s3_keys',
        type=str,
        nargs='+',
        required=True,
        help='Space-separated list of S3 object keys to download.'
    )
    parser.add_argument(
        '--download_paths',
        type=str,
        nargs='+',
        required=True,
        help='Space-separated list of local paths to save the downloaded files.'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Ensure the number of S3 keys matches the number of download paths
    if len(args.s3_keys) != len(args.download_paths):
        print("Error: The number of --s3_keys must match the number of --download_paths.")
        print(f"Received {len(args.s3_keys)} s3_keys and {len(args.download_paths)} download_paths.")
        exit(1)

    # Iterate over each pair of s3_key and download_path
    for s3_key, download_path in zip(args.s3_keys, args.download_paths):
        # Optionally, you can remove the leading '/' from download_path if not desired
        # download_path = download_path.lstrip('/')
        download_file_from_s3(args.bucket_name, s3_key, download_path)

if __name__ == "__main__":
    main()
