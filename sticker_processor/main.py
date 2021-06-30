from face_detector import FaceDetector
from face_processor import FaceProcessor
from animations import ANIMATIONS_LIST
import json
import boto3
import gzip
import redis
import cv2
import numpy as np
import time

session = boto3.session.Session()
redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)

s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net'
)

sqs = boto3.client(
    service_name="sqs",
    endpoint_url='https://message-queue.api.cloud.yandex.net'
)

sqs_url = sqs.get_queue_url(QueueName="to-process")["QueueUrl"]
face_detector = FaceDetector()
face_processor = FaceProcessor("./model_weights/segmentator.pth")


def export_tgs(animation, sanitize=False):
    if sanitize:
        animation.tgs_sanitize()

    lottie_dict = animation.to_dict()
    lottie_dict["tgs"] = 1
    lottie_data = gzip.compress(json.dumps(lottie_dict).encode())
        
    return lottie_data


def get_face_sticker(image):
    face = face_detector.detect(image)
    sticker = face_processor.image2sticker(face)

    return sticker


def get_animated_stickers(face_sticker):
    animated_stickers = []

    for animation in ANIMATIONS_LIST:
        sticker_suffix = animation.get_sticker_name()
        animated_sticker = animation(face_sticker)
        animated_stickers.append((animated_sticker, sticker_suffix))

    return animated_stickers


def upload_animated_stickers(task_id, animated_stickers):
    download_links = []

    for i, (sticker, sticker_suffix) in enumerate(animated_stickers):
        sticker_tgs = export_tgs(sticker)
        sticker_name = f"{task_id}~{i}~{sticker_suffix}.tgs"
        s3.put_object(Bucket='stickers', Key=sticker_name, Body=sticker_tgs, ContentType="application/gzip")

        download_links.append(f"https://storage.yandexcloud.net/stickers/{sticker_name}")

    return download_links


def make_upload_stickers(task_id, image):
    face_sticker = get_face_sticker(image)
    animated_stickers = get_animated_stickers(face_sticker)
    download_links = upload_animated_stickers(task_id, animated_stickers)

    return download_links


def download_image(task_id):
    file_name = task_id + ".jpg"
    s3_object = s3.get_object(Bucket='faces', Key=file_name)
    data = s3_object.get('Body').read()

    image =  cv2.imdecode(np.asarray(bytearray(data)), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def update_task(task_id, status, **kwargs):
    try:
        if redis_client.exists(task_id):
            task = json.loads(redis_client.get(task_id))
        else:
            task = {}

        task["status"] = status
        task.update(**kwargs)
        redis_client.set(task_id, json.dumps(task))

    except Exception as ex:
        print("Task update failed:", ex)

def main():
    print("Procesor is started")
    while True:
        messages = sqs.receive_message(
            QueueUrl=sqs_url,
            MaxNumberOfMessages=10,
            VisibilityTimeout=60,
            WaitTimeSeconds=20
        ).get('Messages')

        if messages is None:
            time.sleep(2)
            continue

        for message in messages:
            task_id = message.get("Body")
            image = download_image(task_id)

            try:
                download_links = make_upload_stickers(task_id, image)
                update_task(task_id, "done", download_links=download_links)
            except Exception as ex:
                print(ex)
            
            sqs.delete_message(
                QueueUrl=sqs_url,
                ReceiptHandle=message.get('ReceiptHandle')
            )


if __name__ == "__main__":
    main()