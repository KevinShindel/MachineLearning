from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from SparkML.service import create_spark


def main():
    spark = create_spark()
    filename = '../../dataset/zoo.csv'
    df = spark.read.format('csv').options(inferSchema=True, header=True).load(filename)
    # TODO: Write logic for mllib classificator
    assembler = VectorAssembler(inputCols=['hair', 'eggs'], outputCol='features')
    vectorized_df = assembler.transform(df).select('features', 'animal_name',)
    for i in range(2, 20):
        k_trainer = KMeans(featuresCol='features', predictionCol='cluster', k=i, seed=i)
        model = k_trainer.fit(vectorized_df)
        evaluator = ClusteringEvaluator()
        predictions = model.transform(vectorized_df)
        silhouette = evaluator.evaluate(predictions)
        print("Silhouette with squared euclidean distance = " + str(silhouette))

    df.show()


if __name__ == '__main__':
    main()
