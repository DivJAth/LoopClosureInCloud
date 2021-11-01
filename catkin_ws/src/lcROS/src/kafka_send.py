
import sys, os
import time
import numpy as np
import lzf
import struct

from producer import *

kafka_producer = connect_kafka_producer()
MAX_LEN = 4294967293 

def decompress(file_name, val_type, val_len, line_len):
    f = open(file_name, 'r')
    byte_array = f.read()

    start = time.time()
    dcomp = lzf.decompress(byte_array, MAX_LEN)
    print 'Decompress time {0}'.format(time.time() - start)
    
    val = struct.unpack(val_type*(len(dcomp) / val_len), dcomp)
    val = list(val[:min(44880,int(len(val)/2244)*2244)])
    print("len before", len(val))
    
    val1 = np.array(val) 
    val1 = np.reshape(val1, (len(val1)/line_len, line_len)) 
    print("val_shape ",val1.shape, type(val1),val1.dtype)
        
    start = time.time()
    np.savez_compressed(r'/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/src/output', data=val1)
    print 'Compress time {0}'.format(time.time() - start)     
    print 'Wrote file {0}.npz in folder {1}.'.format(r'output', r'/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/src/')
    
    print 'Unpack time {0}'.format(time.time() - start)
    return {'val':val, 'h':len(val)/line_len,'w':line_len}

def main(argv):
    features = argv[0]
    feature_len = int(argv[1])
    output_name = argv[2]
    path = r'/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/src/' 
    features_path = path + features         
    data = decompress(features_path , 'd', 8, feature_len)
    # start = time.time()
    # np.savez_compressed(path + '/' + output_name, data=data)
    # print 'Compress time {0}'.format(time.time() - start)     
    # print 'Wrote file {0}.npz in folder {1}.'.format(output_name, path)
    kafka_topic = 'dist_stuff'
    publish_message(kafka_producer, kafka_topic, 'movement', data)
#     # print(type(byte_array))
    return

 

if __name__ == '__main__':
    main(sys.argv[1:])







# import time
# import lzf
# import struct
# from producer import *
# kafka_producer = connect_kafka_producer()

# def mainfunc(lzf_file): 
#     print("size from kafka",len(lzf_file))
#     print(lzf_file[:5])
# #     print("Entered Main Kafka_send_test!")
#     feature_len = 2244
# #     path = r'/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/src/' 
# #     features = r'histo_output.lzf'
# #     features_path = path + features
# #     print("seg here 1")    
# #     try:  
#     data = decompress(lzf_file , 'd', 8, feature_len)
# #         print("seg here 2")
# #     except Exception as ex:
# #         print('Exception during decompression')      
#     kafka_topic = 'dist_stuff'
#     publish_message(kafka_producer, kafka_topic, 'embedding', data)
# #     print("exit Main Kafka_send_test!")
#     return 1

# def decompress(val, val_type, val_len, line_len):
# #     f = open(file_name, 'r')
# #     byte_array = f.read()
#     MAX_LEN = 4294967293 
# #     dcomp = lzf.decompress(byte_array, MAX_LEN)
#     # val = struct.unpack(val_type*(len(dcomp) / val_len), dcomp)
#     val = val[:min(44880,int(len(val)/2244)*2244)]
# #     print("len before", len(val))
#     return {'val':val, 'h':len(val)/line_len,'w':line_len}




 

# # if __name__ == '__main__':
# #     mainfunc("nothing")

