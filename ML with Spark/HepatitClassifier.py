"""
Description: This script is used to classify animals based on their features.
Main goal: Predict if patient is HIP, prediction based on the following columns: Age,
Author: Kevin Shindel
Date: 2024-08-05
"""

# import libraries
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
from pyspark.mllib.evaluation import MulticlassMetrics

from SparkML.service import create_spark


def main():
    spark = create_spark()
    filename = '../dataset/hcvdata.csv'
    df = spark.read.format('csv').options(header=True, inferSchema=True).load(filename).drop('_c0')

    # WorkFlow
    # Task: Predict if patient is HIP
    df.groupby('category').count().show()

    # Feature engineering
    # Numerical values / Vectorization / Scaling

    # convert string Sex to int
    gender_encoder = StringIndexer(inputCol='Sex', outputCol='Gender').fit(df)
    df = gender_encoder.transform(df).drop('Sex')
    df.show()
    print(gender_encoder.labels)

    # convert string Category to int
    category_encoder = StringIndexer(inputCol='Category', outputCol='Target').fit(df)
    df = category_encoder.transform(df).drop('Category')
    df.show()
    print(category_encoder.labels)

    # Revert convert from int to string
    converter = IndexToString(inputCol='Target', outputCol='original_cat')
    transformed_df = converter.transform(df)
    transformed_df.show()

    # Feature selection
    df = df.toPandas().replace('NA', 0).astype(float)
    df = spark.createDataFrame(df)

    features = ['Age', 'Gender', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT', 'Target']
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    vec_df = assembler.transform(df)
    vec_df.show()

    # Train/ Test/ Split
    train_df, test_df = vec_df.randomSplit([0.7, 0.3])
    lr = LogisticRegression(featuresCol='features', labelCol='Target')
    lr_model = lr.fit(train_df)
    y_pred = lr_model.transform(test_df)
    y_pred.select('target', 'rawPrediction', 'probability', 'prediction').show()

    # Model Evaluation
    multi_evaluation = MulticlassClassificationEvaluator(labelCol='Target', metricName='accuracy')
    accuracy = multi_evaluation.evaluate(y_pred)
    print('Accuracy: ', accuracy)

    # Precision, F1 Score, Recall
    lr_metric = MulticlassMetrics(y_pred.select('target', 'prediction').rdd)
    print('Accuracy: ', lr_metric.accuracy)
    print('Precision: ', lr_metric.precision(1.0))
    print('Recall: ', lr_metric.recall(1.0))
    print('F1 Score: ', lr_metric.fMeasure(1.0))

    spark.stop()
    exit(0)


if __name__ == '__main__':
    main()
