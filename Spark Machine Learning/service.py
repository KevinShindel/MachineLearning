""" This module contains the function to create a SparkSession object. """

from pyspark.sql import SparkSession


def create_spark():
    """ Create a SparkSession object. """
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("TestSuite") \
        .config(key='spark.sql.shuffle.partitions', value='4') \
        .config(key='spark.default.parallelism', value='4') \
        .config(key='spark.sql.session.timeZone', value='UTC') \
        .config(key='spark.ui.enabled', value='false') \
        .config(key='spark.app.id', value='Test') \
        .config(key='spark.driver.host', value='localhost') \
        .getOrCreate()

    return spark
