# mahout_HBase_in_pyspark
This code snippet showcases how to use mahout's spark-itemsimilarity API and build item-based recommender engine in Spark's Python API

My spark's version is 1.4.1, Mahout's version is 10.1

The data was generated from Mahout spark-bindings, spark-itemsimilarity API. I generated it from using it through Mahout command line interface :
$MAHOUT_HOME/mahout spark-itemsimilarity -i $hdfshome/data2.csv -o $hdfshome -fc 1 -ic 2 -rc 0 --filter1 view -m 10 -mppu 10

For more info:
https://mahout.apache.org/users/algorithms/intro-cooccurrence-spark.html

If you're gonna run this code snippet, make sure you have add dependecy jars into your spark classpath

These jars should be added in classpath:
/root/spark-1.4.1/examples/target/spark-examples_2.10-1.4.1.jar
/root/spark-1.4.1/examples/target/scala-2.10/spark-examples-1.4.1-hadoop2.5.0.jar
/root/spark-1.4.1/core/target/jars/guava-14.0.1.jar

Simply run pyspark with --jars [classes]