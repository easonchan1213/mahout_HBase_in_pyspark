# coding: utf-8

import os
from test_helper import Test
from pyspark.mllib.linalg import SparseVector


basePath = os.path.join('')
inputFile_itemitemMatrix = os.path.join(basePath,'part-00000')
inputFile_useritemMatrix = os.path.join(basePath,'data2.csv')


if os.path.isfile(inputFile_itemitemMatrix):
    item_item_matrix = (sc
               .textFile(inputFile_itemitemMatrix)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data
    
if os.path.isfile(inputFile_useritemMatrix):
    user_item_matrix = (sc
               .textFile(inputFile_useritemMatrix)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data


# # It's the beginning of the first preparation work : construct a user-item matrix
def parser_uiMatrix(line):
    """Converts a comma separated string into a list of (uid, pid) tuples.

    Args:
        line (str): A comma separated string where the first value is uid and the third is pid.

    Returns:
        tuple: A (uid, pid) tuple.
    """
    linesplit = line.split(',')
    uid = linesplit[0]
    pid = linesplit[2]
    return (uid,pid)


user_itemRDD = user_item_matrix.map(parser_uiMatrix).cache()
user_itemRDDUnique = user_itemRDD.reduceByKey(lambda x,y:x+','+y)


def createMappingDict(inputData):
    """Creates a one-hot-encoder dictionary based on the input data.

    Args:
        inputData (RDD of lists of (int, str)): An RDD of observations where each observation is
            made up of a list of (featureID, value) tuples.

    Returns:
        dict: A dictionary where the keys are item and map to values that are
            unique integers.
    """
    return (inputData
                       .zipWithIndex()
                       .collectAsMap())


readyForOHE_item = user_itemRDD.map(lambda (x,y):y).distinct()
Mapping_item = createMappingDict(readyForOHE_item)
readyForOHE_user = user_itemRDD.map(lambda (x,y):x).distinct()
Mapping_user = createMappingDict(readyForOHE_user)


def computeSparseVector(inputData,MappingDictItem,MappingDictUser):
    """Mapping from raw string RDD into RDD of pyspark.mllib.linalg.sparseVector

    Args:
        inputData (tuple): A tuple of observations, (uid,pid)
        MappingDictItem (dict): A dict where key = pid, value = unique id
        MappingDictUser (dict): A dict where key = uid, value = unique id

    Returns:
        pyspark.mllib.linalg.sparseVector of which row = user, columns = items
    """
    # remove repeating actions
    uid = inputData[0]
    pid = set(inputData[1].split(','))
    
    uidIndex = MappingDictUser[uid]
    pidIndex = sorted([MappingDictItem[p] for p in pid])
    
    # assume all actions are counted the same rating
    length = len(MappingDictItem)
    aLotOf1s = [1.0 for i in pidIndex]
    
    return (uidIndex,SparseVector(length,pidIndex,aLotOf1s))
    

# We have huge advantages if we use sc.broadcast instead of passing mappingDict around
broadcast_Mapping_item = sc.broadcast(Mapping_item)
broadcast_Mapping_user = sc.broadcast(Mapping_user)


user_item_matrix = (user_itemRDDUnique
                    .map(lambda (x,y):computeSparseVector((x,y),broadcast_Mapping_item.value,broadcast_Mapping_user.value))
                    )


# # It's the end of the first preparation work : construct a user-item matrix of sparse vectors
# 
# # Here starts the second preparation work : construct a item-item matrix of sparse vectors

def parser_iiMatrix_and_compute_SparseMatrix(line,MappingDictItem,MappingDictUser):
    """Converts a mahout output record into a list of (item, sparsevector(item)) tuples.

    Args:
        line (str): A string where the first value is item, and the rest are items and their scores

    Returns:
        tuple: a list of (item, sparsevector(item)) tuples.
    """
    linesplit = line.split(',')
    pid = linesplit[0]
    pids = linesplit[1].split(' ')
    
    pidIndex = MappingDictItem[pid]
    pidsIndex = []
    for ele in pids:
        p = ele.split(':')[0]
        p = MappingDictItem[p]
        score = ele.split(':')[1]
        pidsIndex.append((p,score))
    sorted_pidsIndex = sorted(pidsIndex, key=lambda tup: tup[0])
    
    length = len(MappingDictItem)
    sparse_pids = SparseVector(length,[i[0] for i in sorted_pidsIndex],[i[1] for i in sorted_pidsIndex])
    
    return (pidIndex,sparse_pids)

# filter out those records that don't have simialr items
item_item_matrix_haveRatings = item_item_matrix.filter(lambda x:',' in x)
item_item_matrix = (item_item_matrix_haveRatings
                     .map(lambda x:parser_iiMatrix_and_compute_SparseMatrix(x,broadcast_Mapping_item.value,broadcast_Mapping_user.value))   
                    )

def sparseAdd(sv1,sv2,length):
    from pyspark.mllib.linalg import Vectors
    combinedV = sv1.toArray()+sv2.toArray()
    nonzeroes = combinedV.nonzero()[0]
    return Vectors.sparse(length,nonzeroes,combinedV[nonzeroes])

# Test sparseAdd function
o = SparseVector(2241,[771,806,1209,1574],[1.0,1.0,1.0,1.0])
k = SparseVector(2241,[305,1253,1254],[1.0,1.0,1.0])
Test.assertEquals(SparseVector(2241,[305,1253,1254],[2.0,2.0,2.0]),sparseAdd(k,k,k.size),'sparseAdd function malfunc')
Test.assertEquals(SparseVector(2241,[771,806,1209,1574],[2.0,2.0,2.0,2.0]),sparseAdd(o,o,o.size),'sparseAdd function malfunc')
Test.assertEquals(SparseVector(2241,[305,771,806,1209,1253,1254,1574],[1.0,1.0,1.0,1.0,1.0,1.0,1.0]),sparseAdd(o,k,o.size),'sparseAdd function malfunc')


def matrixMultiplication(sv,item_item_matrix):
    indices = sv.indices
#     return item_item_matrix.filter(lambda (x,y):x in indices).map(lambda (x,y):y).reduce(lambda x,y:sparseAdd(x,y,y.size))
    current_sv = None
    result_sv = None
    for i in indices:
        if not current_sv:
            try:
                current_sv = item_item_matrix[i]
                result_sv = current_sv
            except:
                pass
        else:
            result_sv = sparseAdd(current_sv,result_sv,current_sv.size)
    if result_sv:
        return result_sv
    else:
        return SparseVector(1,[],[])
        
def findTopNfromSparseVector(sv,n=10):
    import operator
    
    ind = sv.indices
    val = sv.values
    unsortedDict = {ind[i]:val[i] for i in range(len(ind))}
    sortedList = sorted(unsortedDict.items(),key=operator.itemgetter(1),reverse=True)
    
    return sortedList[:n]
  
# test findTopNfromSparseVector function    
test = SparseVector(2241, {77: 33.8243, 87: 40.1439, 112: 70.4867, 134: 19.6333, 166: 25.6678, 273: 38.1442, 309: 27.909, 394: 32.7246, 464: 31.7367, 549: 27.909, 584: 10.1468, 586: 32.0174, 589: 24.278, 590: 16.7671, 612: 70.4867, 670: 31.7367, 672: 24.278, 696: 32.7246, 765: 32.7246, 775: 30.0197, 862: 59.4238, 937: 33.8243, 976: 24.278, 997: 25.6678, 999: 29.2635, 1037: 31.7367, 1038: 15.8744, 1045: 29.2635, 1052: 29.2635, 1053: 64.1805, 1079: 32.7246, 1097: 520.4584, 1106: 32.7246, 1212: 36.4835, 1232: 65.109, 1254: 4.4758, 1282: 64.1805, 1338: 19.6333, 1339: 35.1785, 1401: 30.0197, 1408: 40.1439, 1484: 32.7246, 1518: 25.6678, 1531: 30.8402, 1537: 24.278, 1576: 40.1439, 1580: 32.7246, 1668: 70.4867, 1687: 38.5564, 1689: 9.005, 1691: 300.6941, 1830: 40.1439, 1839: 24.7189, 1851: 31.7367, 1872: 520.4584, 1940: 40.1439, 1966: 31.7367, 2015: 30.0197, 2026: 31.7367, 2037: 36.4835, 2105: 24.7189})
testOutcome = [(1872, 520.45839999999998),
 (1097, 520.45839999999998),
 (1691, 300.69409999999999),
 (1668, 70.486699999999999),
 (612, 70.486699999999999)]
Test.assertEquals(testOutcome,findTopNfromSparseVector(test,5),'findTopNfromSparseVector function malfunc')


Mapping_from_user = {}
for i in Mapping_user.iterkeys():
    Mapping_from_user[Mapping_user[i]] = i

Mapping_from_item = {}
for i in Mapping_item.iterkeys():
    Mapping_from_item[Mapping_item[i]] = i

Mapping_from_user_broadcast = sc.broadcast(Mapping_from_user)
Mapping_from_item_broadcast = sc.broadcast(Mapping_from_item)


def listToHBaseString(lst,Mapping_fromIndexToItem):
    result = ''
    for ele in lst:
        item = Mapping_fromIndexToItem[ele[0]]
        score = ele[1]
        result += '{0}:{1} '.format(item,score)
    return result[:-1]

def mapFromUserIndex(uidIndex,Mapping_fromIndexToUser):
    return Mapping_fromIndexToUser[uidIndex]

def sparseMultiply(sv,value):
    from pyspark.mllib.linalg import Vectors
    array = sv.toArray() / value
    nonzeroes = array.nonzero()[0]
    return Vectors.sparse(length,nonzeroes,array[nonzeroes])

def normalize(rdd):
    stdev = rdd.map(lambda (x,y):y).flatMap(getValuesFromSV).stdev()
    return rdd.map(lambda (x,y):(x,sparseMultiply(y,stdev)))


itemMatrixCollected = item_item_matrix .collectAsMap()
itemMatrixCollected_broadcast = sc.broadcast(itemMatrixCollected)

finalRDD = (user_item_matrix
            .map(lambda (x,y):(x,matrixMultiplication(y,itemMatrixCollected_broadcast.value)))
            )
normalizedRDD = (normalize(finalRDD)
            .map(lambda (x,y):(x,findTopNfromSparseVector(y,5)))
            .map(lambda (x,y):(mapFromUserIndex(x,Mapping_from_user_broadcast.value),listToHBaseString(y,Mapping_from_item_broadcast.value)))
            )

# #Write into HBase
host = 'master1'
table = 'test'

conf = {"hbase.zookeeper.quorum": host,"hbase.mapred.outputtable": table,"mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat","mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable","mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"


(normalizedRDD
 .map(lambda (x,y):(x,[x,'f1','CF',y]))
 .saveAsNewAPIHadoopDataset(conf=conf,keyConverter=keyConv,valueConverter=valueConv)
 )

