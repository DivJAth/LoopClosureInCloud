
import time
import lzf
import struct
from producer import *
kafka_producer = connect_kafka_producer()

def mainfunc(lzf_file): 
    print("Shperical histogram received by Kafka ",len(lzf_file))
    # print(lzf_file[:5])
    feature_len = 2244
    data = decompress(lzf_file , 'd', 8, feature_len)
    kafka_topic = 'loop_closure'
    publish_message(kafka_producer, kafka_topic, 'embedding', data)
    return 1

def decompress(val, val_type, val_len, line_len):
    MAX_LEN = 4294967293 
    val = val[:int(len(val)/2244)*2244)]
    return {'val':val, 'h':len(val)/line_len,'w':line_len}



