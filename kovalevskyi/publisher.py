import time

from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()

topic_path = publisher.topic_path('going-tfx', 'flightevents')

def callback(message_future):
    if message_future.exception(timeout=30):
        print ('Publishing messsage on {} threw an exception {}'.format(
        'flightevents', message_future.exception()))
    else:
        print(message_future.result())

for n in range (1, 10):
    data = u'Message number {}'.format(n)
    data = data.encode('utf-8')
    message_future = publisher.publish(topic_path, data=data)
    message_future.add_done_callback(callback)

print('Published message IDs:')

while True:
    time.sleep(60)