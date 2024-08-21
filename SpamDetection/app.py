from flask import Flask, render_template, request  # Import necessary Flask modules
import pickle  # Import pickle for loading the model and vectorizer
import matplotlib.pyplot as plt  # Import Matplotlib for creating charts
import seaborn as sns  # Import Seaborn for enhanced data visualization
import io  # Import io for handling in-memory binary streams
import base64  # Import base64 for encoding images to display in HTML
import pandas as pd

app = Flask(__name__)  # Initialize the Flask application

# Load the trained model and vectorizer from disk
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the home route to handle email classification
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None

    # Check if the request method is POST, indicating that form data was submitted
    if request.method == 'POST':
        email_content = request.form['email_content']  # Get email content from the form
        if email_content:  # Ensure that the email content is not empty
            # Transform the email content into the format expected by the model
            email_vector = vectorizer.transform([email_content])
            prediction = model.predict(email_vector)  # Make a prediction
            confidence = model.predict_proba(email_vector).max()  # Calculate the confidence score
            result = "Spam" if prediction[0] == 'spam' else "Ham"  # Determine the result

    # Render the index.html template with the classification result and confidence score
    return render_template('index.html', result=result, confidence=confidence)

# Define a route to display the spam/ham distribution chart
@app.route('/distribution-chart')
def distribution_chart():
    spam_count = 826  # Number of spam emails in the dataset
    ham_count = 4925  # Number of ham emails in the dataset

    labels = ['Spam', 'Ham']  # Labels for the chart
    sizes = [spam_count, ham_count]  # Data for the chart
    colors = ['#ff9999','#66b3ff']  # Colors for the chart segments
    explode = (0.1, 0)  # Explode the first slice for emphasis

    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')  # Ensure the pie chart is circular

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Render the chart.html template with the plot URL
    return render_template('chart.html', plot_url=plot_url)

# Define a route to display the classifier's accuracy report
@app.route('/accuracy-report')
def accuracy_report():
    accuracy = 98.2  # Hardcoded accuracy for demonstration purposes
    return render_template('report.html', accuracy=accuracy)

# Define a route to display the evaluation metrics chart
@app.route('/evaluation-metrics')
def evaluation_metrics():
    # Example data for evaluation metrics
    labels = ['precision', 'recall', 'f1-score']
    spam_scores = [0.95, 0.93, 0.94]  # Replace these with your actual scores
    ham_scores = [0.99, 0.99, 0.99]

    x = range(len(labels))  # Set up the x-axis

    # Create a bar chart to display the evaluation metrics
    fig, ax = plt.subplots()
    ax.bar(x, spam_scores, width=0.4, label='Spam', color='grey', align='center')
    ax.bar(x, ham_scores, width=0.4, label='Not Spam', color='purple', align='edge')

    # Set chart labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Save the plot to a BytesIO object and encode it as a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Render the chart.html template with the plot URL
    return render_template('chart.html', plot_url=plot_url)

@app.route('/email-length-histogram')
def email_length_histogram():
    # Assuming data is already loaded or passed to this function
    data = pd.read_csv('data/email_classification.csv')

    # Generate the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data['email'].apply(len), bins=50, color='blue')
    plt.title('Histogram of Email Lengths')
    plt.xlabel('Length of Emails')
    plt.ylabel('Frequency')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Render the plot in the template
    return render_template('chart.html', plot_url=plot_url)


# Run the Flask application in debug mode (useful for development)
if __name__ == "__main__":
    app.run(debug=True)
