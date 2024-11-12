from flask import Flask, render_template, request
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from plotly.utils import PlotlyJSONEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from collections import Counter
from math import log

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('post_data.csv')

# Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Update the Jinja2 environment to include log
app.jinja_env.globals.update(log=log)

def preprocess_text(text):
    if pd.isna(text):  # Handle None or NaN values
        return ""
    return re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def extract_and_count_hashtags(filtered_data):
    content_strings = filtered_data['Post Content'].apply(lambda x: str(x) if not pd.isna(x) else "")
    hashtags = Counter(re.findall(r"#(\w+)", " ".join(content_strings)))
    return sorted(hashtags.items(), key=lambda item: item[1], reverse=True)

def create_pie_chart(filtered_data):
    sentiment_data = filtered_data['sentiment'].value_counts().reset_index()
    sentiment_data.columns = ['sentiment', 'count']
    
    data = [{
        'type': 'pie',
        'labels': sentiment_data['sentiment'].tolist(),
        'values': sentiment_data['count'].tolist()
    }]
    layout = {
        'title': "Sentiment Distribution"
    }
    
    return {'data': data, 'layout': layout}

def create_time_series(filtered_data):
    if filtered_data.empty:
        return {'data': [], 'layout': {'title': 'No data available'}}

    filtered_data['date'] = pd.to_datetime(filtered_data['Post Timestamp']).dt.date
    sentiment_counts = filtered_data.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    
    # Ensure all sentiment categories are present
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = 0

    sentiment_counts = sentiment_counts.reset_index()

    fig = px.line(sentiment_counts, x='date', y=['Positive', 'Negative', 'Neutral'], title="Sentiment Trends Over Time")
    return {'data': fig.data, 'layout': fig.layout}

def create_sentiment_bar_chart(filtered_data):
    if filtered_data.empty:
        return {'data': [], 'layout': {'title': 'No data available'}}

    # Count sentiment distribution
    sentiment_counts = filtered_data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts['Sentiment'],
            y=sentiment_counts['Count'],
            marker_color=['blue', 'red', 'green']  # You can adjust colors as needed
        )
    ])

    fig.update_layout(
        title='Sentiment Distribution',
        xaxis_title='Sentiment',
        yaxis_title='Count'
    )

    return {'data': fig.data, 'layout': fig.layout}


def create_detailed_hover_line_plot(filtered_data):
    filtered_data_copy = filtered_data.copy()  # Avoid modifying original dataframe
    filtered_data_copy["date"] = pd.to_datetime(filtered_data_copy['Post Timestamp']).dt.date
    
    sentiment_counts = filtered_data_copy.groupby(['date', 'sentiment']).size().reset_index(name='count')
    
    fig = go.Figure()
    
    for sentiment in sentiment_counts['sentiment'].unique():
        df_sentiment = sentiment_counts[sentiment_counts['sentiment'] == sentiment]
        fig.add_trace(go.Scatter(
            x=df_sentiment['date'],
            y=df_sentiment['count'],
            mode='lines+markers',
            name=sentiment,
            hovertemplate=
            '<b>Date</b>: %{x}<br>' +
            '<b>Sentiment</b>: ' + sentiment + '<br>' +
            '<b>Count</b>: %{y}<br>' +
            '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Detailed Sentiment Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="closest"
    )
    
    return {'data': fig.data, 'layout': fig.layout}

def create_interactions_chart(filtered_data):
    indexes = list(range(1, len(filtered_data) + 1))  # Create index for x-axis
    
    fig = go.Figure(data=[
        go.Bar(
            name="Likes", 
            x=indexes, 
            y=filtered_data['Likes Count'], 
            marker_color="blue", 
            customdata=filtered_data['Shortcode'], 
            hovertemplate="Post: %{x}<br>Likes: %{y}<br>Shortcode: %{customdata}<extra></extra>",
        ),
        go.Bar(
            name="Comments", 
            x=indexes, 
            y=filtered_data['Comments Count'], 
            marker_color="red", 
            customdata=filtered_data['Shortcode'], 
            hovertemplate="Post: %{x}<br>Comments: %{y}<br>Shortcode: %{customdata}<extra></extra>",
        )
    ])
    
    fig.update_layout(
        title="Post Interactions",
        xaxis_title="Post Index",
        yaxis_title="Count",
        barmode='group',
        xaxis=dict(
            tickmode='array',
            tickvals=indexes,
            ticktext=[str(i) for i in indexes]
        )
    )
    
    return {'data': fig.data, 'layout': fig.layout}

@app.route("/")
def index():
    usernames = df['Username'].dropna().unique().tolist()
    return render_template('index.html', usernames=usernames)

@app.route('/analyze', methods=['POST'])
def analyze():
    username = request.form['username']
    filtered_data = df[df['Username'] == username].copy()

    if filtered_data.empty:
        return render_template('error.html', error="No data found for the specified username.")

    # Preprocess and analyze sentiment
    filtered_data['clean_content'] = filtered_data['Post Content'].apply(preprocess_text)
    filtered_data['sentiment'] = filtered_data['clean_content'].apply(analyze_sentiment)

    # Generate the pie chart, time series, bar chart, etc.
    graphs = {
        'pie_chart': create_pie_chart(filtered_data),
        'time_series': create_time_series(filtered_data),
        'bar_chart': create_sentiment_bar_chart(filtered_data),
        'detailed_hover_plot': create_detailed_hover_line_plot(filtered_data),
        'interactions_chart': create_interactions_chart(filtered_data)
    }

    # Convert graphs to JSON using Plotly's JSON encoder
    graphs_json = json.dumps(graphs, cls=PlotlyJSONEncoder)

    # Extract hashtags for the hashtag analysis
    sorted_hashtags = extract_and_count_hashtags(filtered_data)

    # Pass the graphs and sorted_hashtags to the results.html template
    return render_template('results.html', graphs_data=graphs_json, sorted_hashtags=sorted_hashtags)

if __name__ == '__main__':
    app.run()
