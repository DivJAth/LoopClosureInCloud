"""#!/usr/bin/env python"""

from kafka import KafkaProducer, KafkaConsumer
# # import numpy as np


def publish_message(producer_instance, topic_name, key, value):
    try:
        key_bytes = bytes(key).encode('utf-8')
        value_bytes = bytes(value).encode('utf-8')

        producer_instance.send(topic_name, key=key_bytes, value=value_bytes)
        producer_instance.flush()
       
        print('Message published successfully.',topic_name)
    except Exception as ex:
        print('Exception in publishing message')
        print(str(ex))


def connect_kafka_producer():
    _producer = None
    try:
        # _producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10),value_serializer=lambda x:dumps(x).encode('utf-8'))
        _producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10))
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(str(ex))
    finally:
        return _producer

# if __name__ == '__main__':
#     kafka_producer = connect_kafka_producer()  
#     publish_message(kafka_producer, key, 'raw', 'message')

#     if kafka_producer is not None:
#         kafka_producer.close()

