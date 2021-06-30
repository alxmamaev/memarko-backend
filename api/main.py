import random
import string
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
import boto3
import json
import redis


session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net'
)
sqs = boto3.client(
    service_name="sqs",
    endpoint_url='https://message-queue.api.cloud.yandex.net'
)

sqs_url = sqs.get_queue_url(QueueName="to-process")["QueueUrl"]
redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


def generate_random_file_name(name_lenght):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(name_lenght))


@app.post("/send_photo")
def send_photo(photo: UploadFile = File(..., media_type="image/jpeg")):
    data = photo.file.read()

    task_id = generate_random_file_name(15)
    file_name = task_id + ".jpg"
    
    task = {"status": "processing"}
    redis_client.set(task_id, json.dumps(task))

    s3.put_object(Bucket='faces', Key=file_name, Body=data, ContentType="image/jpg")
    sqs.send_message(QueueUrl=sqs_url, MessageBody=task_id)

    return {"task_id": task_id}


@app.get("/get_stickers/{task_id}")
def get_stickers(task_id: str):
    if not redis_client.exists(task_id):
        return {"status": "not_exist"}
    else:
        redis_data = json.loads(redis_client.get(task_id))
        return redis_data