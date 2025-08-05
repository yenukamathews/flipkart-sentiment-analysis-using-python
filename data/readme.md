Flipkart Reviews Sentiment Analysis

This project analyzes customer reviews from Flipkart and classifies them as positive or negative using machine learning models.

Objective:

To classify product reviews into positive or negative categories based on the review text.

To use machine learning models for classification after cleaning and processing the data.

To visualize the review sentiment distribution and generate word clouds for insights.

Project Structure:

data/

flipkart_data.csv : Original dataset containing customer reviews and ratings.

notebooks/

sentiment_analysis.py : Main script for data preprocessing, training, evaluation, and visualization.

outputs/

model.pkl : Saved best-performing ML model (Logistic Regression).

visuals/

sentiment_distribution.png : Bar plot of sentiment counts.

positive_wordcloud.png : Word cloud for positive reviews.

negative_wordcloud.png : Word cloud for negative reviews.

README.md : Project overview and documentation.

requirements.txt : Python libraries required to run the project.

Dataset Description:

Dataset file name: flipkart_data.csv

Columns:

review: The actual review text by the customer.

rating: Rating given to the product (1 to 5).

Sentiment derivation:

Sentiment is generated from the rating column.

Ratings >= 3 are considered positive (label = 1).

Ratings < 3 are considered negative (label = 0).

Steps Performed:

Data loading using pandas.

Dropping nulls and duplicates.

Text preprocessing:

Lowercasing.

Removing HTML tags, links, special characters, and punctuation.

Label encoding sentiment from the rating column.

TF-IDF vectorization of cleaned review text.

Splitting data into training and testing sets.

Training multiple models:

Logistic Regression

Multinomial Naive Bayes

Random Forest

Support Vector Machine

Evaluating each model using accuracy and F1 score.

Saving the best-performing model (Logistic Regression) using pickle.

Generating EDA visualizations:

Sentiment distribution bar plot.

Positive and negative review word clouds.

How to Run:

Install dependencies:
pip install -r requirements.txt

Run the main script:
cd notebooks
python sentiment_analysis.py

Output Files:

Model file: outputs/model.pkl

Visuals:

visuals/sentiment_distribution.png

visuals/positive_wordcloud.png

visuals/negative_wordcloud.png

Requirements:

See requirements.txt for all necessary Python libraries.

Author:

Mathews Yenuka

