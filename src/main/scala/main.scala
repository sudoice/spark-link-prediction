import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.GBTClassifier

object Predictor {
  // making this variable global in order to allow "spark.implicits._" be used on every function without re-importing
  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Link Prediction")
    .getOrCreate()

  import spark.implicits._

  // constants
  object Configuration {
    val NODE_INFO_FILENAME = "src/main/resources/node_information.csv"
    val TRAINING_SET_FILENAME = "src/main/resources/training_set.txt"
    val TESTING_SET_FILENAME = "src/main/resources/testing_set.txt"
    val GROUND_TRUTH_FILENAME = "src/main/resources/Cit-HepTh.txt"

    // For the first problem (p1) it is ok to run with value 1, but for the second problem (p2) this value should be
    // maximum 0.2 in order to run under a single machine within a reasonable amount of time.
    val INFO_DATAFRAME_PORTION = 0.1
    val TF_SIZE = 10000
    val LOGISTIC_REGRESSION_ITERATIONS = 100
    val SIMILARITY_THRESHOLD = 0.97
  }

  /**
   * Calculates the difference between two publication years.
   *
   * @param yearFrom the starting year
   * @param yearTo the end year
   * @return an integer indicating the difference.
   */
  def getPublicationYearDifference(yearFrom: Int, yearTo: Int): Int = {
    Math.abs(yearFrom - yearTo)
  }

  /**
   * Checks whether two journal names are the same.
   *
   * @param journalA the first journal
   * @param journalB the second journal
   * @return an integer with value 1 if they are the same or 0 otherwise.
   */
  def isPublishedOnSameJournal(journalA: String, journalB: String): Int = {
    if (journalA == journalB) {
      1
    }
    else {
      0
    }
  }

  /**
   * Counts the common words between two sentences.
   *
   * @param textA the first sentence
   * @param textB the second sentence
   * @return an integer indicating the number of common words.
   */
  def countCommonWords(textA: Seq[String], textB: Seq[String]): Int = {
    if (textA == null || textB == null) {
      0
    }
    else {
      textA.intersect(textB).length
    }
  }

  /**
   * Reads the contents of a CSV file and puts them into a Spark DataFrame.
   *
   * @param sparkSession the currently active Spark session.
   * @return the Spark DataFrame.
   */
  def getInfoDataFrame(sparkSession: SparkSession, filename: String): DataFrame = {
    val columnNames = Seq(
      "srcId",
      "year",
      "title",
      "authors",
      "journal",
      "abstract")

    sparkSession
      .read
      .option("header", "false")
      .csv(filename)
      .toDF(columnNames: _*)
  }

  /**
   * Performs several transformations on the columns of the incoming DataFrame, such as stopwords removal, tokenization,
   * tf-idf calculation
   *
   * @param dataFrame the target DataFrame to process
   * @return the transformed DataFrame
   */
  def preProcess(dataFrame: DataFrame): DataFrame = {
    val abstractTokenizer = new Tokenizer()
      .setInputCol("abstract")
      .setOutputCol("abstract_tokens_raw")

    val abstractStopWordsRemover = new StopWordsRemover()
      .setInputCol("abstract_tokens_raw")
      .setOutputCol("abstract_tokens_clean")

    val titleTokenizer = new Tokenizer()
      .setInputCol("title")
      .setOutputCol("title_tokens_raw")

    val titleStopWordsRemover = new StopWordsRemover()
      .setInputCol("title_tokens_raw")
      .setOutputCol("title_tokens_clean")

    val tf = new HashingTF()
      .setInputCol("abstract_tokens_clean")
      .setOutputCol("tf")
      .setNumFeatures(Configuration.TF_SIZE)

    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tf_idf")

    val transformedDataFrame = dataFrame
      .na
      .fill(Map("abstract" -> "", "title" -> "", "authors" -> "", "journal" -> ""))
      .withColumn("authors_tokens_raw", functions.split(col("authors"), ","))

    val stages = Array(
      abstractTokenizer,
      abstractStopWordsRemover,
      titleTokenizer,
      titleStopWordsRemover,
      tf,
      idf)

    new Pipeline()
      .setStages(stages)
      .fit(transformedDataFrame)
      .transform(transformedDataFrame)
  }

  /**
   * Retrieves the DataFrame that contains the training data.
   *
   * @param sparkContext the current Spark Context
   * @param filename the target filename to load data from
   * @return
   */
  def getTrainingDataFrame(sparkContext: SparkContext, filename: String): DataFrame = {
    sparkContext
      .textFile(filename)
      .map(line => {
        val fields = line.split(" ")

        (fields(0), fields(1), fields(2).toInt)
      })
      .toDF("srcId", "dstId", "label")
  }

  /**
   * Retrieves the DataFrame that contains the testing data.
   *
   * @param sparkContext the current Spark Context
   * @param filename the target filename to load data from
   * @return
   */
  def getTestingDataFrame(sparkContext: SparkContext, filename: String): DataFrame = {
    sparkContext
      .textFile(filename)
      .map(line => {
        val fields = line.split(" ")

        (fields(0), fields(1))
      })
      .toDF("srcId", "dstId")
  }

  /**
   * Retrieves the DataFrame that contains the ground truth data.
   *
   * @param sparkContext the current Spark Context
   * @param filename the target filename to load data from
   * @return
   */
  def getGroundTruthDataFrame(sparkContext: SparkContext, filename: String): DataFrame = {
    sparkContext
      .textFile(filename)
      .map(line => {
        val fields = line.split("\t")

        (fields(0), fields(1))
      })
      .toDF("srcId", "dstId")
  }

  /**
   * Joins the training and information DataFrames into one, so each row contains the "from" and "to" information about
   * two nodes among with their label.
   *
   * @param trainingDataFrame the training DataFrame
   * @param infoDataFrame the information DataFrame
   * @return the joined DataFrame
   */
  def joinDataFrames(trainingDataFrame: DataFrame, infoDataFrame: DataFrame): DataFrame = {
    val joinedDataFrame = trainingDataFrame
      .as("a")
      // the <=> operator means "equality test that is safe for null values"
      .join(infoDataFrame.as("b"), $"a.srcId" <=> $"b.srcId")
      .select($"a.srcId",
        $"a.dstId",
        $"a.label",
        $"b.year",
        $"b.title_tokens_clean",
        $"b.authors_tokens_raw",
        $"b.journal",
        $"b.abstract_tokens_clean")
      .withColumnRenamed("srcId", "id_from")
      .withColumnRenamed("dstId", "id_to")
      .withColumnRenamed("year", "year_from")
      .withColumnRenamed("title_tokens_clean", "title_from")
      .withColumnRenamed("authors_tokens_raw", "authors_from")
      .withColumnRenamed("journal", "journal_from")
      .withColumnRenamed("abstract_tokens_clean", "abstract_from")
      .as("a")
      .join(infoDataFrame.as("b"), $"a.id_to" <=> $"b.srcId")
      .withColumnRenamed("year", "year_to")
      .withColumnRenamed("title_tokens_clean", "title_to")
      .withColumnRenamed("authors_tokens_raw", "authors_to")
      .withColumnRenamed("journal", "journal_to")
      .withColumnRenamed("abstract_tokens_clean", "abstract_to")
      .drop("srcId")

    joinedDataFrame
  }

  /**
   * Prepares the incoming DataFrame for binary classification by combining the required feature columns into one.
   *
   * @param joinedDataFrame the join of training and information DataFrames.
   * @return the final DataFrame with the additional column "features".
   */
  def getFinalDataFrame(joinedDataFrame: DataFrame): DataFrame = {
    val commonTitleWords = udf(countCommonWords(_: Seq[String], _: Seq[String]))
    val commonAuthors = udf(countCommonWords(_: Seq[String], _: Seq[String]))
    val commonAbstractWords = udf(countCommonWords(_: Seq[String], _: Seq[String]))
    val isSameJournal = udf(isPublishedOnSameJournal(_: String, _: String))
    val publicationYearDifference = udf(getPublicationYearDifference(_: Int, _: Int))
    val toDouble = udf((i: Int) => if (i == 1) 1.0 else 0.0)

    val finalDataFrame = joinedDataFrame
    .withColumn("common_title_words", commonTitleWords(joinedDataFrame("title_from"), joinedDataFrame("title_to")))
    .withColumn("common_authors", commonAuthors(joinedDataFrame("authors_from"), joinedDataFrame("authors_to")))
    .withColumn("common_abstract_words", commonAbstractWords(joinedDataFrame("abstract_from"), joinedDataFrame("abstract_to")))
    .withColumn("publication_year_difference", publicationYearDifference(joinedDataFrame("year_from"), joinedDataFrame("year_to")))
    .withColumn("is_same_journal", isSameJournal(joinedDataFrame("journal_from"), joinedDataFrame("journal_to")))
    .withColumn("label", toDouble(joinedDataFrame("label")))
    .select("label",
      "common_title_words",
      "common_authors",
      "common_abstract_words",
      "publication_year_difference",
      "is_same_journal",
      "tf_idf")

    val assembler = new VectorAssembler()
      .setInputCols(Array("common_title_words",
        "common_authors",
        "common_abstract_words",
        "publication_year_difference",
        "is_same_journal",
        "tf_idf"))
      .setOutputCol("features")

    assembler
      .transform(finalDataFrame)
      .na
      .drop()
  }

  /**
   * Retrieves the testing DataFrame and after joining with the ground truth DataFrame, adds the values 0 or 1 to it as
   * labels in order to be easily evaluated.
   *
   * @param testingDataFrame the testing DataFrame
   * @param groundTruthDataFrame the ground truth DataFrame
   * @return
   */
  def addLabelsToTestDataFrame(testingDataFrame: DataFrame, groundTruthDataFrame: DataFrame): DataFrame = {
    val labeledTestingDataFrame = testingDataFrame
      .as("a")
      .join(
        groundTruthDataFrame
          .as("b"),
          $"a.srcId" <=> $"b.srcId" &&
          $"a.dstId" <=> $"b.dstId",
          "left"
      )
      .withColumn("label", when($"b.srcId".isNull, 0).otherwise(1))
      .drop($"b.srcId")
      .drop($"b.dstId")

    labeledTestingDataFrame
  }

  /**
  * Calculates and prints the metrics for the predictions of a binary classification model like SVM
  * that doesn't provide probability scores.
  *
  * @param predictions the DataFrame containing the predictions.
  */
  def calculateMetricsSVM(predictions: DataFrame): Unit = {
    // Create RDD of (prediction, label) pairs for binary metrics
    val predictionAndLabels = predictions
      .select("prediction", "label")
      .rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))
    
    // Count for confusion matrix and derived metrics
    val tp = predictions.filter(col("prediction") === 1.0 && col("label") === 1.0).count()
    val fp = predictions.filter(col("prediction") === 1.0 && col("label") === 0.0).count()
    val tn = predictions.filter(col("prediction") === 0.0 && col("label") === 0.0).count()
    val fn = predictions.filter(col("prediction") === 0.0 && col("label") === 1.0).count()
    
    // Calculate metrics for positive class (class 1)
    val precision1 = if (tp + fp > 0) tp.toDouble / (tp + fp) else 0.0
    val recall1 = if (tp + fn > 0) tp.toDouble / (tp + fn) else 0.0
    val f1Score1 = if (precision1 + recall1 > 0) 2 * precision1 * recall1 / (precision1 + recall1) else 0.0
    
    println("\nMetrics for positive class:")
    println(s"Precision: $precision1")
    println(s"Recall: $recall1")
    println(s"F1-Score: $f1Score1")
    
    // Calculate metrics for negative class (class 0)
    val precision0 = if (tn + fn > 0) tn.toDouble / (tn + fn) else 0.0
    val recall0 = if (tn + fp > 0) tn.toDouble / (tn + fp) else 0.0
    val f1Score0 = if (precision0 + recall0 > 0) 2 * precision0 * recall0 / (precision0 + recall0) else 0.0
    
    println("\nMetrics for negative class:")
    println(s"Precision: $precision0")
    println(s"Recall: $recall0")
    println(s"F1-Score: $f1Score0")
    
    // Print confusion matrix
    println("\nConfusion Matrix:")
    println(s"TP: $tp, FP: $fp")
    println(s"FN: $fn, TN: $tn")
    
    // Overall accuracy
    val accuracy = (tp + tn).toDouble / (tp + tn + fp + fn)
    println(s"\nOverall Accuracy: $accuracy")
    
    // For SVM, we don't have probabilistic outputs for ROC curve, but we can still use
    // the BinaryClassificationMetrics with binary predictions for basic evaluation
    try {
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)
      println(s"\nArea under ROC curve: ${metrics.areaUnderROC()}")
      println(s"Area under PR curve: ${metrics.areaUnderPR()}")
    } catch {
      case e: Exception => 
        println("\nCouldn't calculate ROC/PR curves - SVM outputs binary predictions only.")
        println("To get ROC curves, use a probabilistic classifier like LogisticRegression.")
    }
  }

  /**
   * Calculates and prints the metrics for the predictions of a binary classification model.
   *
   * @param predictions the DataFrame containing the predictions.
   */
  def calculateMetrics(predictions: DataFrame): Unit = {
    // Using probability scores instead of binary predictions
    val predictionAndLabels = predictions
      .select("label", "probability")
      .rdd
      .map(row => {
        // Extract probability of positive class (class 1)
        val probability = row.getAs[org.apache.spark.ml.linalg.Vector]("probability").toArray(1)
        (probability, row.getAs[Double]("label"))
      })
    
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    
    // Print AUC
    println(s"Area under ROC curve: ${metrics.areaUnderROC()}")
    
    // Get F1-Score by threshold
    val f1ScoreByThreshold = metrics.fMeasureByThreshold().collect()
    
    // Find threshold that gives the best F1 score
    val bestF1Threshold = f1ScoreByThreshold.maxBy(_._2)
    val optimalThreshold = bestF1Threshold._1
    val bestF1Score = bestF1Threshold._2
    
    println(s"\nOptimal threshold based on best F1-Score: $optimalThreshold")
    println(s"Best F1-Score: $bestF1Score")
    
    // Find precision and recall at the best threshold
    val precisionAtOptimalThreshold = metrics.precisionByThreshold()
      .filter(_._1 == optimalThreshold)
      .first()._2
    
    val recallAtOptimalThreshold = metrics.recallByThreshold()
      .filter(_._1 == optimalThreshold)
      .first()._2
    
    // For positive class (class 1)
    println("\nMetrics for positive class at optimal threshold:")
    println(s"Precision: $precisionAtOptimalThreshold")
    println(s"Recall: $recallAtOptimalThreshold")
    
    // For negative class (class 0) - need to compute from confusion matrix
    // Create UDF to extract probability of class 1
    val getProb1 = udf((v: org.apache.spark.ml.linalg.Vector) => v.toArray(1))
    
    val predictionsWithThreshold = predictions
      .withColumn("prob1", getProb1(col("probability")))
      .withColumn("prediction_optimal", 
        when(col("prob1") >= optimalThreshold, 1.0).otherwise(0.0))
    
    val tp = predictionsWithThreshold.filter(col("prediction_optimal") === 1.0 && col("label") === 1.0).count()
    val fp = predictionsWithThreshold.filter(col("prediction_optimal") === 1.0 && col("label") === 0.0).count()
    val tn = predictionsWithThreshold.filter(col("prediction_optimal") === 0.0 && col("label") === 0.0).count()
    val fn = predictionsWithThreshold.filter(col("prediction_optimal") === 0.0 && col("label") === 1.0).count()
    
    // For negative class, precision = TN/(TN+FN), recall = TN/(TN+FP)
    val precision0 = if (tn + fn > 0) tn.toDouble / (tn + fn) else 0.0
    val recall0 = if (tn + fp > 0) tn.toDouble / (tn + fp) else 0.0
    val f1Score0 = if (precision0 + recall0 > 0) 2 * precision0 * recall0 / (precision0 + recall0) else 0.0
    
    println("\nMetrics for negative class at optimal threshold:")
    println(s"Precision: $precision0")
    println(s"Recall: $recall0")
    println(s"F1-Score: $f1Score0")
    
    // Print confusion matrix
    println("\nConfusion Matrix at optimal threshold:")
    println(s"TP: $tp, FP: $fp")
    println(s"FN: $fn, TN: $tn")
    
    // Overall accuracy
    val accuracy = (tp + tn).toDouble / (tp + tn + fp + fn)
    println(s"\nOverall Accuracy at optimal threshold: $accuracy")
  }

  /**
   * Problem 1:
   * Given the network and a list of possible links, provide predictions if the links exist or not.
   *
   * @param sparkContext the current Spark Context
   */
  def p1(sparkContext: SparkContext, mn: Int): Unit = {
    println("Retrieving DataFrames...")
    val infoDataFrame = preProcess(getInfoDataFrame(spark, Configuration.NODE_INFO_FILENAME)
      .sample(Configuration.INFO_DATAFRAME_PORTION, 12345L))
    val trainingDataFrame = getTrainingDataFrame(sparkContext, Configuration.TRAINING_SET_FILENAME)
    val testingDataFrame = getTestingDataFrame(sparkContext, Configuration.TESTING_SET_FILENAME)
    val groundTruthDataFrame = getGroundTruthDataFrame(sparkContext, Configuration.GROUND_TRUTH_FILENAME)
    val labeledTestingDataFrame = addLabelsToTestDataFrame(testingDataFrame, groundTruthDataFrame)

    println("Joining DataFrames...")
    val joinedTrainDataFrame = joinDataFrames(trainingDataFrame, infoDataFrame)
    val joinedTestDataFrame = joinDataFrames(labeledTestingDataFrame, infoDataFrame)

    val finalTrainDataFrame = getFinalDataFrame(joinedTrainDataFrame)
    val finalTestDataFrame = getFinalDataFrame(joinedTestDataFrame)
    val model = if(mn == 0) {
      println("Running Logistic Regression classification...\n")
      new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setProbabilityCol("probability")
        .setRawPredictionCol("prediction_raw")
        .setMaxIter(Configuration.LOGISTIC_REGRESSION_ITERATIONS)
    } else if(mn == 1) {
      println("Running Naive Bayes classification...\n")
      new NaiveBayes()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setProbabilityCol("probability")
        .setModelType("multinomial")
        .setSmoothing(1.0)
    } else if(mn == 2) {
      println("Running Linear SVM classification...\n")
      new LinearSVC()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMaxIter(10)
        .setRegParam(0.1)
        .setTol(1e-4)
    } else {
      println("Running Gradient Boosted Trees classification...\n")
      new GBTClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setProbabilityCol("probability")
        .setMaxIter(2)
        .setMaxDepth(2)
        .setStepSize(0.1)
        .setMaxBins(32)
        .setMinInstancesPerNode(1)
        .setMinInfoGain(0.0)
        .setSubsamplingRate(0.8)
        .setSeed(1234L)
    }

    val predictions = model
      .fit(finalTrainDataFrame)
      .transform(finalTestDataFrame)

    println("Calculating metrics...\n")
    if(mn == 2){
      calculateMetricsSVM(predictions)
    } else {
      calculateMetrics(predictions)
    }
  }
  def main(args: Array[String]): Unit = {
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")
    p1(sparkContext, 0)
    spark.stop()
  }
}