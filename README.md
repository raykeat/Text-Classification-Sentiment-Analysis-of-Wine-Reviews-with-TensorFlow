# Text Classification/Sentiment Analysis of Wine Reviews with TensorFlow

This project aims to carry out text classification or sentiment analysis of wine reviews using TensorFlow.

# Main Software and Frameworks Used
1. JupyterLab / Google Colab was used as the development environment.
2. Pandas and Numpy were used for construction of datasets and data analysis/manipulation.
3. Matplotlib was used for data visualization.
4. BeautifulSoup was used for web scraping to retrieve 18 wine blog posts/articles from www.vinography.com.
5. TensorFlow was used as the Machine Learning Model to carry out binary classification of wine reviews.

# Methodology

## Data Collection, Data Preprocessing and Feature Engineering
1. The labelled training data was obtained from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/). The dataset contained a description for each wine review, as well as a corresponding score (0-100) assigned for each review.
2. Data preprocessing was carried out to remove rows with missing values, and exploratory data analysis carried out with matplotlib to determine range of scores and their frequencies.
3. Feature Engineering carried out to create a new binary output label column with a label of 1 for scores>=90, and label of 0 for scores<90.
4. The description was selected as the input feature, and binary output label was selected as the output label.
5. Pandas Dataframe with these 2 columns was split into training, validation and test datasets in 8:1:1 ratio, and then converted to a TensorFlow Dataset Object. (to prepare data in format that can be used efficiently by TensorFlow, especially when working with textual data in the case of NLP)

## Model Training and Evaluation
6. 2 Different TensorFlow Models were used.
   
   a. The first was a pretrained text embedding model from tensorflow hub (nnlm-en-dim50/2: https://tfhub.dev/google/nnlm-en-dim50/2)
      1. The pretrained text embedding layer was used as the First Keras layer, followed by a few dense layers and a final layer with an output node and sigmoid function.
      2. The text embedding layer converts textual data to vectors of numerical values in a way that captures the semantic meaning of the text.
      3. Model was then trained on training dataset with Adam Optimizer, Binary Cross Entropy Loss Function, with accuracy and recall metrics.
      4. Model was evaluated on test dataset with accuracy of 0.8279 and recall of 0.7796.
   
   b. The second was a Long Short Term Memory (LSTM) Recurrent Neural Network (RNN).
      1. The first layer is a Text Encoder/Text Vectorization Layer that converts raw text into integer sequences by tokening the text and mapping tokens to integers.
      2. The second layer is a Embedding Layer that converts integer sequences into vector representations.
      3. This is followed by 2 LSTM Bidirectional Layers, 1 Dense layer and a final layer with 1 output node to predict output label.
      4. Model was trained on training dataset with Adam Optimizer, Binary Cross Entropy Loss Function, with accuracy and recall metrics.
      5. Model was evaluated on test dataset with accuracy of 0.8341.
   
7. The LSTM Model was chosen due to higher accuracy.

## Web Scraping Wine Articles from vinography.com to predict classification labels for them
8. 18 Wine Blogs Posts/Articles were scraped from vinography.com.
9. Articles were written into a csv file "wine_articles.csv".
10. Pandas Dataframe was created by reading csv file, and dataframe was converted to tensorflow dataset object.

## Using Trained LSTM Model to generate sentiment analysis of wine articles
11. The previously trained LSTM RNN Model was used to predict the output probabilities (of belonging to class 1/having positive sentiment) for each article.
12. The results were shown in a pandas dataframe and converted to excel csv file "results.csv"





