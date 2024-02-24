import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import plotly.express as px
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def extract_playlist_id(youtube_link):
    pattern = r'list=([^&]+)'
    match = re.search(pattern, youtube_link)
    if match:
        playlist_id = match.group(1)
        return playlist_id
    else:
        return None

def get_all_video_ids_from_playlists(youtube, playlist_ids):
    all_videos = []
    for playlist_id in playlist_ids:
        next_page_token = None
        while True:
            playlist_request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            try:
                playlist_response = playlist_request.execute()
                all_videos += [item['contentDetails']['videoId'] for item in playlist_response['items']]
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            except HttpError as e:
                st.error(f"Error fetching playlist with ID {playlist_id}: {e}")
                break
    return all_videos

# Function to get replies for a specific comment
def get_replies(youtube, parent_id, video_id):  # Added video_id as an argument
    replies = []
    next_page_token = None

    while True:
        reply_request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        )
        reply_response = reply_request.execute()

        for item in reply_response['items']:
            comment = item['snippet']
            replies.append({
                'Timestamp': comment['publishedAt'],
                'Username': comment['authorDisplayName'],
                'VideoID': video_id,
                'Comment': comment['textDisplay'],
                'Date': comment['updatedAt'] if 'updatedAt' in comment else comment['publishedAt'],
                'Likes': comment.get('likeCount', 0)
            })

        next_page_token = reply_response.get('nextPageToken')
        if not next_page_token:
            break

    return replies

# Function to get all comments (including replies) for a single video
def get_comments_for_video(youtube, video_id):
    all_comments = []
    next_page_token = None

    while True:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            textFormat="plainText",
            maxResults=100
        )
        comment_response = comment_request.execute()

        for item in comment_response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'Timestamp': top_comment['publishedAt'],
                'Username': top_comment['authorDisplayName'],
                'VideoID': video_id,  # Directly using video_id from function parameter
                'Comment': top_comment['textDisplay'],
                'Date': top_comment['updatedAt'] if 'updatedAt' in top_comment else top_comment['publishedAt'],
                'Likes': top_comment.get('likeCount', 0)
            })

            # Fetch replies if there are any
            if item['snippet']['totalReplyCount'] > 0:
                all_comments.extend(get_replies(youtube, item['snippet']['topLevelComment']['id'], video_id))

        next_page_token = comment_response.get('nextPageToken')
        if not next_page_token:
            break

    return all_comments

if __name__ == "__main__":
    st.sidebar.title("🎥YouTube Playlist Sentiment Analysis💟")
    st.sidebar.markdown("Gain a deeper understanding of the sentiment behind the comments on your favorite YouTube playlists.")
    st.sidebar.markdown("Try our **YouTube Playlist Sentiment Analysis** tool now and uncover the emotional landscape of your videos!")
    st.sidebar.markdown("### 💡How to Use:")
    st.sidebar.markdown("1. **Enter your YouTube API key🔑** for authentication.")
    api_key = st.sidebar.text_input("Enter your YouTube API key:", type="password")

    st.sidebar.markdown("2. **Input the link to the YouTube playlist🎬** you want to analyze.")
    playlist_link = st.sidebar.text_input("Enter YouTube Playlist Link:")
    playlist_ids = extract_playlist_id(playlist_link)

    st.sidebar.markdown("3. **Hit the Enter💪** button.")
    st.sidebar.button("Enter")

    st.sidebar.markdown("4. **Explore sentiment distribution📊** through dynamic charts and discover the top positive and negative comments within the playlist.")
    st.sidebar.markdown("**Credits**")
    st.sidebar.markdown("- https://www.youtube.com/watch?app=desktop&v=A1III_DQU4I")
    st.sidebar.markdown("- https://www.youtube.com/watch?app=desktop&v=uwLWf0rEL18")
    st.sidebar.markdown("Made with ❤️ by Karla Kovacs")

    if not playlist_link:
        pass
    else:
        progress_bar = st.progress(0)

        youtube = build('youtube', 'v3', developerKey=api_key)
        video_ids = get_all_video_ids_from_playlists(youtube, [extract_playlist_id(playlist_link)])

        all_comments = []
        
        for i, video_id in enumerate(video_ids):
            progress_percentage = (i + 1) / len(video_ids)
            progress_bar.progress(progress_percentage)
            video_comments = get_comments_for_video(youtube, video_id)
            all_comments.extend(video_comments)

        comments_df = pd.DataFrame(all_comments)
        comments_df = comments_df.drop(['VideoID','Date'],axis=1)

        nltk.download('vader_lexicon')
        sentiments = SentimentIntensityAnalyzer()
        comments_df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in comments_df["Comment"]]
        comments_df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in comments_df["Comment"]]
        comments_df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in comments_df["Comment"]]
        comments_df['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in comments_df["Comment"]]
        score = comments_df["Compound"].values
        sentiment = []
        for i in score:
            if i >= 0.05:
                sentiment.append('Positive')
            elif i <= -0.05:
                sentiment.append('Negative')
            else:
                sentiment.append('Neutral')
        comments_df["Sentiment"] = sentiment

        st.success("The process is finished!")
        playlist_request = youtube.playlists().list(
                part='snippet',
                id=playlist_ids
            )
        playlist_response = playlist_request.execute()
        playlist_title = playlist_response['items'][0]['snippet']['title']
        st.title(f"Playlist: {playlist_title}")
        
        st.subheader("Head of the DataFrame📈")
        st.dataframe(comments_df.head())

        sentiment_counts = comments_df['Sentiment'].value_counts()
        labels = sentiment_counts.index
        values = sentiment_counts.values
        
        st.subheader("Sentiment Counts💗")
        sentiment_counts = comments_df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        st.subheader("Chronological Timeline of Comments by Sentiment📅")
        comments_df['Timestamp'] = pd.to_datetime(comments_df['Timestamp'])
        sorted_comments = comments_df.sort_values(by='Timestamp')
        fig = px.scatter(sorted_comments, x='Timestamp', y='Sentiment', color='Sentiment', labels={'Timestamp': 'Time', 'Sentiment': 'Sentiment'})
        st.plotly_chart(fig)
        
        st.subheader("Select the number of comments to display💬")
        n_comments = st.slider("Number of comments", min_value=1, max_value=10)
        
        st.subheader(f"Top {n_comments} Most Positive Comments👍")
        most_positive_comments = comments_df.nlargest(n_comments, 'Compound')['Comment']
        for i, comment in enumerate(most_positive_comments, start=1):
            st.markdown(f"**Comment {i}**")
            st.markdown(f"{comment}")

        st.subheader(f"Top {n_comments} Most Negative Comments👎")
        most_negative_comments = comments_df.nsmallest(n_comments, 'Compound')['Comment']
        for i, comment in enumerate(most_negative_comments, start=1):
            st.markdown(f"**Comment {i}**")
            st.markdown(f"{comment}")

        st.subheader(f"Most Liked {n_comments} Comments❤️")
        sorted_comments = comments_df.sort_values(by='Likes', ascending=False)
        top_comments_df = sorted_comments.head(n_comments)[['Comment', 'Likes']]
        st.dataframe(top_comments_df.head(n_comments))

        st.subheader("Export the Data into a CSV⬇️")
        csv_export = comments_df.to_csv(index=False, encoding='utf-8')
        st.download_button(label="Download CSV", data=csv_export, file_name='data.csv', key='download_csv')
