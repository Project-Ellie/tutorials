import time
from google.cloud import pubsub_v1


subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('going-tfx', 'test_subscription')

def callback(message):
    print('Received message: {}'.format(message))
    message.ack()
    
subscriber.subscribe(subscription_path, callback=callback)

print('Listening for messages on {}'.format(subscription_path))

while True: 
    time.sleep(10)