import pandas as pd  # Importing pandas for data manipulation
from sklearn.model_selection import train_test_split, GridSearchCV  # Importing tools for splitting data and hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TF-IDF Vectorizer for text data transformation
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Importing metrics for evaluating the model
from imblearn.over_sampling import SMOTE  # Importing SMOTE for handling class imbalance
import seaborn as sns  # Importing Seaborn for data visualization
import matplotlib.pyplot as plt  # Importing Matplotlib for creating plots
import pickle  # Importing pickle for saving and loading models

def load_data():
    """
    Load the dataset from a CSV file and display the first few rows.
    """
    data = pd.read_csv('data/email_classification.csv')  # Loading the dataset
    print(data.head())  # Displaying the first few rows of the dataset for inspection
    return data  # Returning the loaded data

def preprocess_data(data):
    """
    Preprocess the data by splitting it into training and test sets, vectorizing the text data,
    and handling class imbalance using SMOTE.
    """
    # Extracting features (email content) and labels (spam or ham)
    X = data['email']
    y = data['label']

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizing the text data using TF-IDF (Unigrams and Bigrams)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Handling class imbalance in the training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, vectorizer  # Returning the processed data and vectorizer

def train_model(X_train, y_train):
    """
    Train the Logistic Regression model with hyperparameter tuning using GridSearchCV.
    """
    # Defining the Logistic Regression model
    model = LogisticRegression()

    # Defining the hyperparameters to tune
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'solver': ['liblinear', 'saga']  # Optimization algorithms
    }

    # Performing hyperparameter tuning with 5-fold cross-validation
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=5)
    grid.fit(X_train, y_train)  # Training the model

    print("Best parameters found: ", grid.best_params_)  # Displaying the best parameters
    return grid.best_estimator_  # Returning the best model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using accuracy, confusion matrix, and classification report.
    """
    # Predicting the test set results
    y_pred = model.predict(X_test)

    # Evaluating the model's performance
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Displaying the accuracy
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  # Displaying the confusion matrix
    print("Classification Report:\n", classification_report(y_test, y_pred))  # Displaying the classification report

def descriptive_statistics(data):
    """
    Calculate and display descriptive statistics such as mean and median email length,
    and the distribution of spam vs. ham emails.
    """
    # Calculating mean and median email length
    mean_length = data['email'].apply(len).mean()
    median_length = data['email'].apply(len).median()
    print(f"Mean email length: {mean_length}")
    print(f"Median email length: {median_length}")

    # Displaying the distribution of spam vs. ham emails
    print("Distribution of labels:\n", data['label'].value_counts())

def create_visualizations(data):
    """
    Create visualizations for the class distribution and email length distribution.
    """
    # Creating a bar plot for the distribution of spam vs. ham emails
    sns.countplot(x='label', data=data)
    plt.title('Distribution of Spam vs. Ham Emails')
    plt.show()

    # Creating a histogram for the distribution of email lengths
    data['email_length'] = data['email'].apply(len)
    plt.hist(data['email_length'], bins=50, color='blue')
    plt.title('Histogram of Email Lengths')
    plt.xlabel('Length of Emails')
    plt.ylabel('Frequency')
    plt.show()

def main():
    """
    Main function to execute the steps: Load data, preprocess, train the model, evaluate,
    and save the model and vectorizer.
    """
    data = load_data()  # Load the data

    # Perform descriptive statistics and create visualizations
    descriptive_statistics(data)
    create_visualizations(data)

    # Preprocess the data and train the model
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model and vectorizer for future use
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

if __name__ == "__main__":
    main()  # Run the main function when the script is executed
