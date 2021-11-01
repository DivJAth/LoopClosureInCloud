from lshash import LSHash

lsh = LSHash(32, 8)
lsh = LSHash(6, 8, 1, {"redis":{"host": 'localhost', "port": 6379}})
 

lsh.index([1, 2, 3, 4, 5, 6, 7, 8])
lsh.index([2, 3, 4, 5, 6, 7, 8, 9])
lsh.index([10, 12, 99, 1, 5, 31, 2, 3])
lsh.query([1, 2, 3, 4, 5, 6, 7, 7])
print(lsh.query([1, 2, 3, 4, 5, 6, 7, 7]))

# # another method
# lsh.index([[1, 2, 3, 4, 5, 6, 7, 8],
#            [2, 3, 4, 5, 6, 7, 8, 9],
#            [4, 2, 3, 1, 5, 6, 7, 8],
#            [1, 3, 4, 5, 6, 7, 8, 9],
#            [10, 12, 99, 1, 5, 31, 2, 3]])
# print(lsh.query([1, 2, 3, 4, 5, 6, 7, 7], distance_func='jaccard'))