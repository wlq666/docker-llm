from minio import Minio
from minio.error import S3Error
import random
from datetime import timedelta
import json

def getRandom(randomlength=10):
    digits = "0123456789"
    ascii_letters = "abcdefghigklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    str_list = [random.choice(digits + ascii_letters) for i in range(randomlength)]
    random_str = ''.join(str_list)
    print(random_str)
    return random_str

def upload_file(source_file):
    client = Minio("124.193.167.71:19000",
        access_key="zGVaTSxqrhz6bPuVfSls",
        secret_key="SvMqa7zY1xJAw2dxKUZtKKsvECOQu9cExYjYXcXx",
        secure=False
    )

    bucket_name = "agicoin"
    destination_file = 'output_' + getRandom(20) + '.' + source_file.split('.')[-1]
    
    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    result = client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name, result.etag, result.version_id,
        ),
    )
    url = client.get_presigned_url(
        "GET",
        bucket_name,
        result.object_name,
        expires=timedelta(minutes=30),
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )
    print("url:", url)
    return url


