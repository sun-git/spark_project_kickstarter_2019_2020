package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.functions.length
import org.apache.spark.sql.functions.round
import org.apache.spark.sql.functions.lower
import org.apache.spark.sql.functions.datediff
import org.apache.spark.sql.functions.from_unixtime

object Preprocessor {

  def main(args: Array[String]): Unit = {


    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")
    //    println("\n========df.show==========")
    //    df.show()
    println("\n========df.printschema=============")
    df.printSchema()

    import spark.implicits._
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))
    println("\n=========after change df.printschema============")
    dfCasted.printSchema()

    val df2: DataFrame = dfCasted.drop("disable_communication")

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    println("\n=========show country === False ============")
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    println("\n=========clean final_status ============")
    def cleanFinalStatus(final_status: Int): Int = {
      if (final_status == 1 )
        1
      else
        0
    }
    val cleanFinalStatusUdf = udf(cleanFinalStatus _)
    val dfFinal_status = dfCountry
      .withColumn("final_status2", cleanFinalStatusUdf($"final_status"))
      .drop("final_status")
      .withColumnRenamed("final_status2", "final_status")

    dfFinal_status
      .groupBy("final_status").count.orderBy($"count".desc).show(100)

    println("\n=========days_campaign ============")

    val dfDaysCampaign: DataFrame = dfFinal_status
      .withColumn("deadline2", from_unixtime($"deadline") )
      .withColumn("created_at2", from_unixtime($"created_at") )
      .withColumn("launched_at2", from_unixtime($"launched_at") )
      .withColumn("days_campaign", datediff($"deadline2", $"launched_at2"))

    println("\n=========hours_prepa ============")
    val dfHoursPrepa = dfDaysCampaign
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600, 3))

    val dfDropDate: DataFrame = dfHoursPrepa
      .drop("launched_at", "created_at", "deadline")

    println("\n=========lower ============")

    val dfLower: DataFrame = dfDropDate
      .withColumn("name2", lower($"name"))
      .withColumn("desc2", lower($"desc"))
      .withColumn("keywords2", lower($"keywords"))

    println("\n=========colonne text ============")
    import org.apache.spark.sql.functions.concat_ws
    val dfText: DataFrame = dfLower
      .withColumn("text", concat_ws(" ", $"name2", $"desc2", $"keywords2"))

    println("\n=========Valeurs nulles date clean============")
    def cleanDateNull(input: java.lang.Integer): java.lang.Integer = {  // there is no null if input is "Int"
      if (input == null )
        -1
      else
        input
    }
    def cleanHourNull(input: java.lang.Float ): java.lang.Float = {  // there is no null if input is "Float"
      if (input == null )
        -1
      else
        input
    }

    val cleanDateNullUdf = udf(cleanDateNull _)
    val cleanHourNullUdf = udf(cleanHourNull _)

    val dfDateCleanNull: DataFrame = dfText
      .withColumn("days_campaign", cleanDateNullUdf($"days_campaign"))
      .withColumn("hours_prepa", cleanHourNullUdf($"hours_prepa"))
      .withColumn("goal", cleanDateNullUdf($"goal"))
      .filter($"days_campaign" !== -1)
      .filter($"hours_prepa" !== -1)
      .filter($"goal" !== -1)

    println("\n=========Valeurs nulles currency2 country2 clean============")
    def cleanCountryNull(country: String): String = {
      if (country == null || country.length != 2)
        "unknown"
      else
        country
    }
    def cleanCurrencyNull(currency: String): String = {
      if (currency == null)
        "unknown"
      else
        currency
    }

    val cleanCountryNullUdf = udf(cleanCountryNull _)
    val cleanCurrencyNullUdf = udf(cleanCurrencyNull _)

    val dfCountryCleanNull: DataFrame = dfDateCleanNull
      .withColumn("country2", cleanCountryNullUdf($"country2"))
      .withColumn("currency2", cleanCurrencyNullUdf($"currency2"))
      .filter($"country2" !== "unknown")
      .filter($"currency2" !== "unknown")

    dfCountryCleanNull.groupBy("country2").count.orderBy($"count".desc).show(100)

    dfCountryCleanNull.printSchema()
//    dfCountryCleanNull.show()
    dfCountryCleanNull.groupBy("country2").count.orderBy($"count".desc).show(100)

    println("\n=========save data============")
    dfCountryCleanNull.write.mode("overwrite").parquet("src/main/resources/preprocessed")
  }
}
