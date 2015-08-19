# mahout_HBase_in_pyspark
This code snippet showcases how to use mahout's spark-itemsimilarity API and build item-based recommender engine in Spark's Python API

My spark's version is 1.4.1, Mahout's version is 10.1

The data was generated from Mahout spark-bindings, spark-itemsimilarity API. I generated it from using it through Mahout command line interface :
$MAHOUT_HOME/mahout spark-itemsimilarity -i $hdfshome/data2.csv -o $hdfshome -fc 1 -ic 2 -rc 0 --filter1 view -m 10 -mppu 10

For more info:
https://mahout.apache.org/users/algorithms/intro-cooccurrence-spark.html