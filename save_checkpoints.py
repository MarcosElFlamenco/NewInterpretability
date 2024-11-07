import boto3
import torch


def empty_s3_bucket(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.objects.all().delete()


def checkpoint_exists(bucket_name, checkpoint_key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=checkpoint_key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise

def download_checkpoint(bucket_name, checkpoint_key, local_file_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, checkpoint_key, local_file_path)

def upload_checkpoint(local_file_path, bucket_name, checkpoint_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_file_path, bucket_name, checkpoint_key)

def save_checkpoint(epoch, model, optimizer, bucket_name, checkpoint_key):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint.pth')
    upload_checkpoint('checkpoint.pth', bucket_name, checkpoint_key)

def load_checkpoint(bucket_name, checkpoint_key, device):
    local_file_path = 'checkpoint.pth'
    if checkpoint_exists(bucket_name, checkpoint_key):
        print('A checkpoint does in fact exist, now loading...')
        download_checkpoint(bucket_name, checkpoint_key, local_file_path)
        checkpoint = torch.load(local_file_path, map_location=device)
        return checkpoint
    else:
        return None

