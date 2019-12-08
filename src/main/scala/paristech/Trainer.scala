package paristech

import org.apache.spark.SparkConf

import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.NaiveBayes

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n========df.read data=============")

    val df = spark.read.parquet("src/main/resources/preprocessed")
    //    df.show()
    df.printSchema()


    println("\n========tokenizer transform=============")
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    println("\n========StopWordsRemover=============")
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("token_filtered")


    println("\n========la partie TF CountVectorizer=============")
    val countVectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("wordFeatures")


    println("\n========la partie IDF=============")
    val idf = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")

    println("\n========country2 en quantités numériques=============")
    val indexerCountry2 = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    println("\n========currency2 en quantités numériques=============")
    val indexerCurrency2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    println("\n========One-Hot encoder ces deux catégories=============")
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array(indexerCountry2.getOutputCol, indexerCurrency2.getOutputCol))
      .setOutputCols(Array("country_onehot", "currency_onehot"))


    println("\n========assembler tous les features en un unique vecteur=============")
    val assembler = new VectorAssembler()
      .setInputCols(Array(idf.getOutputCol, "days_campaign", "hours_prepa", "goal") ++ encoder.getOutputCols)
      .setOutputCol("features")

    println("\n========LogisticRegression=============")
    import org.apache.spark.ml.classification.LogisticRegression
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf,
        indexerCountry2, indexerCurrency2, encoder, assembler, lr))

    println("\n========Linear Support Vector Machine=============")
    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")

    val pipelineLsvc = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf,
        indexerCountry2, indexerCurrency2, encoder, assembler, lsvc))

    println("\n========Naive Bayes=============")
    val nb = new NaiveBayes()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")

    val pipelineNaiveBayes = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf,
        indexerCountry2, indexerCurrency2, encoder, assembler, nb))

    //    println("\n========Pipeline model with all the data=============")
    //    val model = pipeline.fit(df)
    //    val predict_output=model.transform(df)
    //    predict_output.select("features", "final_status", "predictions", "raw_predictions"
    //      ).show(false)
    //    predict_output.groupBy("final_status", "predictions").count.show()

    println("\n========Split train and test=============")
    val Array(dfTrain, dfTest) = df.randomSplit(Array(0.9, 0.1),seed=0L)

    val modelTrain = pipeline.fit(dfTrain)
    val dfWithSimplePredictions = modelTrain.transform(dfTest)
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    val f1Eval = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val f1Score = f1Eval.evaluate(dfWithSimplePredictions)
    println("\n f1 score Logistic regression: " + f1Score)
    println()

    val modelTrainLsvc = pipelineLsvc.fit(dfTrain)
    val dfWithSimplePredictionsLsvc = modelTrainLsvc.transform(dfTest)
    dfWithSimplePredictionsLsvc.groupBy("final_status", "predictions").count.show()

    val f1EvalLsvc = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val f1ScoreLsvc = f1EvalLsvc.evaluate(dfWithSimplePredictionsLsvc)
    println("\n f1 score Linear Support Vector Machine: " + f1ScoreLsvc)
    println()

    val modelTrainNaiveBayes = pipelineNaiveBayes.fit(dfTrain)
    val dfWithSimplePredictionsNaiveBayes = modelTrainNaiveBayes.transform(dfTest)
    dfWithSimplePredictionsNaiveBayes.groupBy("final_status", "predictions").count.show()

    val f1EvalNaiveBayes = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val f1ScoreNaiveBayes = f1EvalNaiveBayes.evaluate(dfWithSimplePredictionsNaiveBayes)
    println("\n f1 score NaiveBayes: " + f1ScoreNaiveBayes)
    println()

    println("\n========Grid search for Logistic regression=============")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVectorizer.minDF, Array(55.0, 75.0, 95.0))
      .build()

    val f1Grid = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(f1Grid)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val modelGrid = trainValidationSplit.fit(dfTrain)
    val dfWithPredictions = modelGrid.transform(dfTest)
    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    val f1EvalGrid = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
    val f1ScoreGrid = f1EvalGrid.evaluate(dfWithPredictions)
    println("\n f1 score with Grid Search for Logistic Regression: " + f1ScoreGrid)
    modelGrid.write.overwrite().save("src/main/resources/model")
  }
}
