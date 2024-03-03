import streamlit as st
from transformers import pipeline

# Create a sentiment analysis pipeline for multilingual sentiment analysis
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-multilingual-cased')

def analyze_sentiment(text):
    # Analyze the sentiment of the input text
    result = sentiment_analyzer(text)
    label = result[0]['label']
    
    # Convert label to sentiment string
    sentiment = "Positive" if label == 'LABEL_0' else "Negative"
    score = result[0]['score']
    
    return sentiment, score

def main():
    st.title("Multilingual Sentiment Analysis App")
    
    # Create a text input field for the user to enter text for sentiment analysis
    user_input = st.text_area("Enter text for sentiment analysis:")
    
    # Check if the user has entered any text
    if user_input:
        # Analyze the sentiment of the user's input using the sentiment_analyzer pipeline
        sentiment, score = analyze_sentiment(user_input)
        
        # Display the sentiment and score on the Streamlit web app
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Score: {score}")

if __name__ == "__main__":
    main()
