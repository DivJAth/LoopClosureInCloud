# import sys
from json import loads
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark import SparkConf
from pyspark.streaming.kafka import KafkaUtils
from math import pow, atan2, sqrt, radians, sin, cos
from pymongo import MongoClient

import sys, os
import time
import numpy as np
import lzf
import struct

from embedding import *
import time
import datetime

import redis
from lshash import LSHash

# lsh = LSHash(32, 200) 
lsh = LSHash(32, 200,1,{"redis":{"hostname":"localhost", "port":"6539"}}) 


def train(data): 
    print("Calculate Embeddings \n")
    x_anchor = sess.run([hidden5_anchor], feed_dict={x_a: data})[0]
    print(x_anchor.shape)
    return x_anchor


def handler(message):
    records = message.collect()
    print("Length of collection: ",len(records))
    for record in records:
        output = train(record)
        output = output.flatten()
        print("\n Correspondance calculation")
        results =lsh.query(output, num_results=5, distance_func="euclidean")
        if len(results) > 0 :
                print("results seen before \n")
        else:
            print("\n new points")
        lsh.index(output)
        print("similarity check done \n")
        
def decode(msg):
    data = eval(msg[1])
    val = np.array(data['val'])
    val = np.reshape(val, (data['h'], data['w']))
    print("Decode done, val_shape ",val.shape, type(val),val.dtype)
    print("\n")
    return val

if __name__ == "__main__":
    sc=SparkContext(appName="PythonkafkaWordCount")
    ssc=StreamingContext(sc,10)
    sc.setLogLevel("OFF")
    brokers,topic1=sys.argv[1:]
    i=0
    while i< 10:
        print(i)
        i+=1

    kvs=KafkaUtils.createStream(ssc,brokers,"Spark-streaming-consumer",{topic1:1})
    kvs.map(decode).foreachRDD(handler)
    ssc.start()
    ssc.awaitTermination()
