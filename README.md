# Spark project MS Big Data Télécom : Kickstarter campaigns

Spark project for MS Big Data Telecom based on Kickstarter campaigns 2019-2020

To hand in the TP, I only updated the source code in `src/main/scala/paristech/` and data in `src/main/resources/train/`. The `build_and_submit.sh` has changed in my own PC to suit for my own path and these "build" files are omitted to reduce the size.  


## Data cleaning 

For data cleaning part in `src/main/scala/paristech/Preprocessor.scala`, the source is in `src/main/resources/train/train_clean.csv`, and the output will be in `src/main/resources/preprocessed`. 

It uses the `UDF`, `drop`, `filer` and some spark sql functions to do the cleaning and transformation of the data.  



## Model Training

The model training part is in `src/main/scala/paristech/Trainer.scala`, data source is the output of the prreceding step in `src/main/resources/preprocessed`.

For this project, logistic regression model is mainly used and finally the best model with the grid research is saved in `src/main/resources/model`.

I also compared the F1 score with the classification model of Linear Support Vector Machine and Naive Bayes.

### Result

From the results, we can see that the Linear Support Vector Machine has the best result similar to the result of logistic regression with grid search. The Naive Bayes has the worst result.

The grid research for Linear SVM and Naive Bayes is not added at last because it requires a lot of computing capacity and crashed sometimes in my own pc.

#### Logistic regression


|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 1733|
|           0|        1.0| 2301|
|           1|        1.0| 1655|
|           0|        0.0| 5127|



 f1 score Logistic regression: 0.6340436224084591

#### Linear Support Vector Machine


|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 2511|
|           0|        1.0|  814|
|           1|        1.0|  877|
|           0|        0.0| 6614|



 f1 score Linear Support Vector Machine: 0.6569861463182625

#### Naive Bayes


|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0|  726|
|           0|        1.0| 4237|
|           1|        1.0| 2662|
|           0|        0.0| 3191|



 f1 score NaiveBayes: 0.5484452395812549


#### Grid search for Logistic regression


|final_status|predictions|count|
|------------|-----------|-----|
|           1|        0.0| 1032|
|           0|        1.0| 2860|
|           1|        1.0| 2356|
|           0|        0.0| 4568|



 f1 score with Grid Search for Logistic Regression: 0.6531431589452619