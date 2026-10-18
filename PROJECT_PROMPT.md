# Project Prompt: NewsCheck AI — Fake News Detection and News Source Verification System

Act as a professional AI/ML engineer, full-stack developer, and UI/UX designer. Build a complete, functional college project called "NewsCheck AI: An AI-Powered Fake News Detection and Source Verification System."

## 1. Project Objective

Develop a web application that allows users to enter their own news headline, news article, WhatsApp forward, or social media claim. The system should analyze the submitted news and search for supporting or contradicting information from established news organizations such as The Hindu, The Times of India, BBC News, Reuters, and other accessible credible sources.

The system must help users understand whether a claim is supported, contradicted, or still unverified based on available evidence. Do not classify news as fake merely because no matching article is found.

## 2. Technology Stack

Use the following technologies:

- Frontend: HTML5, CSS3, JavaScript
- Backend: Python and Flask
- Machine learning: Scikit-learn
- Natural Language Processing: TF-IDF and text preprocessing
- ML algorithms: Logistic Regression and Passive-Aggressive Classifier
- News search: A permitted news-search API or search service
- Data processing: Pandas and NumPy
- Model storage: Joblib
- Database: SQLite for prediction history
- Charts: Chart.js

Use free or low-cost tools wherever possible. Never hardcode fake API results or pretend a live search was performed.

## 3. Main Features

### A. News Input
- Allow users to paste their own headline or complete article.
- Include a text area and an Analyze News button.
- Validate empty or excessively short input.
- Optionally allow users to provide the original article URL.

### B. AI-Based News Classification
- Train an ML model using a labeled fake-and-real news dataset.
- Preprocess news text using NLP techniques.
- Convert text into numerical features using TF-IDF.
- Predict whether the text resembles fake or real news in the training dataset.
- Clearly explain that the model prediction is not proof of truth or falsity.

### C. Live News Source Verification
Search for relevant articles and reports using an authorized API or search provider.

Include sources such as:
- The Hindu
- The Times of India
- BBC News
- Reuters
- Other relevant credible news outlets
- Official government websites when the claim concerns a government announcement

For each matching article, display:
- News source name
- Article headline
- Publication date, if available
- Short relevant excerpt, if permitted
- Original article URL
- A link to open the source

Use the source's actual search results. Do not invent article titles, publication dates, URLs, or quotations.

If a source cannot be searched, show that it was unavailable rather than claiming it was checked.

### D. Evidence-Based Verification Engine

Implement a separate verification component that:
1. Extracts the main factual claim from the user's input.
2. Searches for related reports.
3. Compares the claim with available article content.
4. Checks relevant dates, names, locations, amounts, and other factual details.
5. Identifies evidence that supports or contradicts the claim.
6. Distinguishes conflicting information from missing information.

Display one of these results:

- Supported by available evidence
- Contradicted by available evidence
- Mixed or conflicting evidence
- Unverified — insufficient evidence

Explain why the result was assigned and link to the evidence.

Do not use the ML classification alone to decide whether a real-world claim is true. Do not treat a newspaper's silence as evidence that the claim is false.

### E. User Dashboard

Create an attractive dashboard containing:
- Total news submissions analyzed
- Number of claims supported by evidence
- Number of claims contradicted by evidence
- Number of unverified claims
- Recent analysis history
- Model performance metrics

Clearly distinguish dataset-based model predictions from evidence-based verification results.

### F. Analysis History

Store each submission and its analysis timestamp in SQLite. Allow users to view previous analyses and their source links. Include a clear-history option and avoid collecting unnecessary personal information.

### G. Model Evaluation

Evaluate the trained classifier using separate training and testing data.

Display:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Prevent data leakage and use appropriate train/test splitting. Report the actual measured results instead of inventing performance percentages.

## 4. UI/UX Design

Create a modern, responsive, professional website with a dark navy and teal color scheme.

Design the following pages or sections:

1. Home page with project introduction
2. News analysis page with a large text input
3. Results page with evidence and source links
4. Dashboard with statistics and charts
5. History page
6. About and methodology page

Include:
- Clean navigation bar
- Responsive cards
- Loading animation while analysis is running
- Clear error messages
- Color-coded result labels with text descriptions
- Clickable source links
- Mobile-friendly layout
- Professional typography and spacing

The interface should resemble a real AI-powered fact-checking platform, not a basic student form.

## 5. Dataset and Model Training

Use a publicly available labeled fake-and-real news dataset, such as the Kaggle Fake and Real News Dataset, subject to its license and suitability.

Create a separate train_model.py script that:
- Loads the dataset
- Cleans and validates the data
- Preprocesses the text
- Splits the dataset into training and testing sets
- Trains and evaluates the classifiers
- Saves the best model and its fitted vectorizer using Joblib

Document the dataset limitations. Ensure that duplicate or near-duplicate articles do not artificially inflate test performance where practical.

## 6. Suggested Project Structure

Create this structure:

FakeNewsDetector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── dataset/
├── database/
├── services/
│   ├── news_search.py
│   └── verification.py
├── templates/
│   ├── index.html
│   ├── results.html
│   ├── dashboard.html
│   ├── history.html
│   └── about.html
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   └── app.js
    └── images/

Modify this structure if a better architecture is necessary.

## 7. Security and Reliability

- Store API keys in environment variables.
- Never expose API keys in frontend JavaScript.
- Add error handling, timeouts, and input validation.
- Respect API rate limits, website terms, and copyright.
- Do not scrape restricted websites without permission.
- Show source links and timestamps when available.
- Clearly communicate when live verification fails.
- Never present an AI prediction as guaranteed truth.
- Do not generate fake search results or unsupported citations.

## 8. Expected Deliverables

Provide:
1. Complete working source code for all required files.
2. Installation commands for Windows.
3. A requirements.txt file.
4. Instructions to obtain and configure any API key.
5. Instructions to download and place the dataset.
6. Commands to train the model.
7. Commands to start the Flask application.
8. Instructions for testing each feature.
9. A README with setup and troubleshooting steps.
10. A project architecture diagram and explanation suitable for a college presentation.

## 9. Development Instructions

Build the project step by step, but deliver complete, consistent code for every required file. Do not provide incomplete snippets or omit essential files.

Start with the folder structure and installation instructions. Then implement model training, news searching, evidence comparison, the Flask backend, frontend, database, and testing.

Make sure the project can run locally on a Windows laptop using VS Code. If a live news API is unavailable, implement a clear fallback mode that reports the limitation honestly.

The final result should be a functional, attractive, academically presentable Fake News Detection and Source Verification System.