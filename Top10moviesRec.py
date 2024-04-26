import sqlite3
import pandas as pd
import csv
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from langchain_openai import ChatOpenAI
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import numpy as np

# Initialize the chat model
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Connect to SQLite database
conn = sqlite3.connect('data/ml-latest-small/movilens_small.db')

# Load data from tables into DataFrames
movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)

# Merge ratings with movies
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

num_of_users=len(ratings_df['userId'].unique())

print("Number of Users:")
print(num_of_users)

unique_genres = set(genre for sublist in movies_df['genres'].str.split('|') for genre in sublist)

# Print the unique genres
print("Unique Genres:", len(unique_genres))
print(unique_genres)

# Specify the user's desired genre(s)
desired_genres_input = input("Enter desired genres (comma-separated, leave empty for any genre): ")
desired_genres = desired_genres_input.split(',') if desired_genres_input else []

# Get movies in the specified genre(s)
if desired_genres:
    genre_movies = movies_df[movies_df['genres'].apply(lambda x: any(genre in x.split('|') for genre in desired_genres))]['movieId'].tolist()
else:
    genre_movies = movies_df['movieId'].tolist()


# Filter ratings for movies in the desired genre(s)
filtered_ratings = merged_df[merged_df['movieId'].isin(genre_movies)]

# Create a Surprise Dataset
reader = Reader(rating_scale=(0.5, 5))  # Adjust the rating scale if needed
data = Dataset.load_from_df(filtered_ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Initialize the SVD algorithm
algo = SVD()

# Train the model
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Evaluate the model (optional)
accuracy.rmse(predictions)

# Get a list of all movie IDs and their corresponding titles and genres
movie_id_title_map = movies_df.set_index('movieId')[['title', 'genres']].to_dict(orient='index')


##############################################################################################################################
# Top 10 movies liked by the user 
user_input = input("Enter a single user ID or a range of IDs (e.g., 100-110): ")

# Parse user input to extract user IDs
if '-' in user_input:
    start_id, end_id = map(int, user_input.split('-'))
    user_ids = list(range(start_id, end_id + 1))
else:
    user_ids = [int(user_input)]

csv_file='results/evaluation.csv'

# Iterate over each user ID and perform the evaluation
for user_id in user_ids:

    print(f"Performing evaluation for user ID: {user_id}")

    user_watched_movies = merged_df[merged_df['userId'] == user_id]['movieId'].tolist()

    # Create a defaultdict to store movie scores
    movie_scores = defaultdict(float)

    # Calculate scores for unwatched movies based on predicted ratings and other factors
    for movie_id in movie_id_title_map.keys():
        if movie_id not in user_watched_movies and movie_id in genre_movies:
            predicted_rating = algo.predict(user_id, movie_id).est
            
            # You can add other factors here
            
            # Calculate the final score
            final_score = predicted_rating
            
            movie_scores[movie_id] = final_score

    # Sort movie scores in descending order
    sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the titles, genres, and scores of top k recommended movies
    # k = int(input("Enter the number of recommendations you want (k): "))  # Specify k
    k=10
    top_movie_scores = sorted_movie_scores[:k]

    # Get the titles, genres, and scores of top k recommended movies
    top_movies_with_scores_and_genres = [(movie_id_title_map[movie_id]['title'], movie_id_title_map[movie_id]['genres'], score) for movie_id, score in top_movie_scores]

    top_movies_recommended_output = "\n Top Recommended Movies with Scores and Genres: \n"
    top_movies_recommended_output_without_score_information = "\n"
    for movie, genres, score in top_movies_with_scores_and_genres:
        top_movies_recommended_output_without_score_information += f"Movie: {movie} - Genres: {genres}\n"
        top_movies_recommended_output += f"Movie: {movie} - Genres: {genres} - Score: {score:.2f} \n"
        
    print(top_movies_recommended_output)

    # Assuming top_movies_with_scores_and_genres is a list of tuples containing movie name, genres, and score
    # Assuming you also have a rank for each movie in the list

    # Initialize an empty list to store movie information
    movie_info_list_RS = []

    # Iterate through each movie in top_movies_with_scores_and_genres
    for rank, (movie, genres, score) in enumerate(top_movies_with_scores_and_genres, start=1):
        # Create a dictionary for each movie containing name, rank, and rating
        score_formatted = f"{score:.2f}"
        movie_info = {'Movie': movie, 'Rank': rank, 'Rating': score_formatted}
        movie_info_list_RS.append(movie_info)

    print("Final List Generated by RS")
    # Print the list of movie information
    for movie_info in movie_info_list_RS:
        print(movie_info)
    ##########################################################################################################################
    # Movies the user has liked previosuly either in the same genre or his top movies of all times.

    # Get the user's top 5 movies and their ratings in the same genre if available
    user_genre_ratings = defaultdict(list)
    for index, row in merged_df[merged_df['userId'] == user_id].iterrows():
        genres = row['genres'].split('|')
        for genre in genres:
            user_genre_ratings[genre].append((row['title'], row['rating']))

    user_top_genre_movies = []
    for genre in desired_genres:
        user_top_genre_movies.extend(user_genre_ratings[genre])
        user_top_genre_movies.sort(key=lambda x: x[1], reverse=True)

    # Initialize an empty list to store the final top movies
    top_movies_output = []

    # If there are movies in the same genre, add them to the top movies list
    if user_top_genre_movies:
        top_movies_output.append("Top Movies Watched by User in Same Genre:")
        for movie, rating in user_top_genre_movies[:5]:
            genres = ', '.join([genre for genre in movies_df[movies_df['title'] == movie]['genres'].iloc[0].split('|')])
            top_movies_output.append(f"Movie: {movie} - Genres: {genres} - Rating: {rating}")

    # If not enough movies in the same genre, add top movies of all time
    if len(top_movies_output) < 5:
        user_top_movies = merged_df[merged_df['userId'] == user_id].sort_values(by='rating', ascending=False).head(5-len(top_movies_output)+1)
        top_movies_output.append("\nTop Movies Watched by User in General:")
        for index, row in user_top_movies.iterrows():
            genres = ', '.join([genre for genre in movies_df[movies_df['title'] == row['title']]['genres'].iloc[0].split('|')])
            top_movies_output.append(f"Movie: {row['title']} - Genres: {genres} - Rating: {row['rating']}")

    # Format the output into a printable string
    top_movies_output_str = "\n".join(top_movies_output)

    # Print the formatted output
    print(top_movies_output_str)


    #######################################################################################################################
    # Now evaluating with the chat model.

    chat_genres= "/n".join(desired_genres)

    top_movies_prompt= f"\nImagine you are a movie recommendation system. There is a user who wants to watch movies in the genres: \t {chat_genres}. Here are 5 movies the user has watched before for some reference and history:\n {top_movies_output_str}. I want you to describe each of these movies breifly. Tell me their plot, characters, and the reason why the A. General Audience likes these movie and B. Why our user likes these movies \n"

    print("\n\n Input Prompt A: \n", top_movies_prompt)    

    top_movies_prompt_output = chat_model.invoke(top_movies_prompt)

    print("\n\n Output Prompt A: \n",top_movies_prompt_output.content)

    

    user_preferences_prompt = f"\n Based on your answer in \t {top_movies_prompt_output.content} \t, what do you think are the user preferences. What kind of plots or characters is the user most likely to be resonate with? Does our user get influenced by the audience, meaning do you think they have matching views or the user would have been influenced by the general public? \n"

    user_preferences_prompt_output = chat_model.invoke(user_preferences_prompt)

    print("\n\n Input Prompt B: \n",user_preferences_prompt)

    print("\n\n Output Prompt B: \n",user_preferences_prompt_output.content)
    
    chat_model_evaluation= f"\n Imagine you are a movie recommender system. The user wants to watch a movie in the \t {chat_genres} \t genres. Based on your thoughts about the user in \t {user_preferences_prompt_output.content} \t, which of the following movies do you think the user will prefer watching: \n {top_movies_recommended_output_without_score_information}? \n Give me a Yes or No answer with a breif explanation for each movie. \n"

    chat_model_evaluation_output= chat_model.invoke(chat_model_evaluation)

    print("\n\n Input Prompt C: \n",chat_model_evaluation)

    print("\n\n Output Prompt C: \n",chat_model_evaluation_output.content)

    chat_model_ranking = f"\n Imagine you are a movie recommender system. Previously you provided me these answers in \t {top_movies_prompt_output.content}, \n {user_preferences_prompt_output.content} and \n {chat_model_evaluation_output.content}. \n Based on your answers how do you think the user will rank the {top_movies_recommended_output_without_score_information} \t, and give what rating will the user give them. The rating can be upto two decimal points from 0-5. Give a brief of reasoning for each movie rating and ranking.\n"

    chat_model_ranking_output= chat_model.invoke(chat_model_ranking)

    print("\n\n Input Prompt D: \n",chat_model_ranking)

    print("\n\n Output Prompt D: \n",chat_model_ranking_output.content)

    

    chat_model_final_ranking = f"\n Based on the ratings in \t {chat_model_ranking_output.content} \t, give me an ordered list of rank, movie name, and rating the user will provide. The list should be from highest rating to the lowest rating. It should strictly follow the format Rank: , Movie: , Rating: \n"

    chat_model_final_ranking_output= chat_model.invoke(chat_model_final_ranking)

    print("\n\n Input Prompt E: \n",chat_model_final_ranking)

    print("\n\n Output Prompt E: \n",chat_model_final_ranking_output.content)


    # Initialize an empty list to store movie information
    movie_info_list_LLM = []

    # Use regular expressions to extract rank, movie name, and rating for each movie
    # movie_pattern = r"Rank: (\d+), Movie: (.*), Rating: (\d+\.\d+)"
    movie_pattern = r"Rank: (\d+), Movie: (.*?), Rating: (\d+(?:\.\d+)?)"

    matches = re.findall(movie_pattern, chat_model_final_ranking_output.content)

    # Iterate through the matches and create a dictionary for each movie
    for match in matches:
        rank = match[0]
        movie_name = match[1]
        rating = match[2]
        
        # Create a dictionary for each movie containing rank, movie name, and rating
        movie_info = { 'Movie': movie_name, 'Rank': rank, 'Rating': rating}
        movie_info_list_LLM.append(movie_info)

    ###################################################################################################################
    #  Final Lists for comparisions

    print("\n Final List Generated by RS")
    # Print the list of movie information
    for movie_info in movie_info_list_RS:
        print(movie_info)


    print("\n Final List Generated by LLM")
    # Print the list of movie information
    for movie_info in movie_info_list_LLM:
        print(movie_info)

    #########################################################################################################################
    # Trying Evaluation of the output
    # Calculate MAE and RMSE for rating discrepancies and overlap percentage for Rankings.
    ratings_RS = [float(movie['Rating']) for movie in movie_info_list_RS]
    ratings_LLM = [float(movie['Rating']) for movie in movie_info_list_LLM]

    sorted_movie_info_list_RS = sorted(movie_info_list_RS, key=lambda x: x['Movie'])
    sorted_movie_info_list_LLM = sorted(movie_info_list_LLM, key=lambda x: x['Movie'])

    # Extract ratings from sorted lists
    ratings_RS = np.array([float(movie['Rating']) for movie in sorted_movie_info_list_RS])
    ratings_LLM = np.array([float(movie['Rating']) for movie in sorted_movie_info_list_LLM])

    # Calculate RMSE and MAE for ratings
    rmse_ratings = np.sqrt(mean_squared_error(ratings_RS, ratings_LLM))
    mae_ratings = mean_absolute_error(ratings_RS, ratings_LLM)

    # Sort the lists by movie names
    sorted_movie_info_list_RS = sorted(movie_info_list_RS, key=lambda x: x['Movie'])
    sorted_movie_info_list_LLM = sorted(movie_info_list_LLM, key=lambda x: x['Movie'])

    # Extract rankings from sorted lists
    rankings_RS = np.array([int(movie['Rank']) for movie in sorted_movie_info_list_RS])
    rankings_LLM = np.array([int(movie['Rank']) for movie in sorted_movie_info_list_LLM])

    # Calculate overlap percentage for rankings
    overlap_percentage_rankings = np.mean(rankings_RS == rankings_LLM) * 100

    # Print the results
    print("\n \n Sorted RS Ratings:", ratings_RS) 
    print("Sorted LLM Ratings:", ratings_LLM)
    print("RMSE for Ratings:", rmse_ratings) 
    print("MAE for Ratings:", mae_ratings) 
    print("Overlap Percentage for Rankings:", overlap_percentage_rankings)



    ##########################################################################################################################
    # Alternatively ca also consider doing their analysis like this
    
    mad_ratings = np.mean(np.abs(ratings_RS - ratings_LLM))

    # Calculate Root Mean Squared Difference (RMSD)
    rmsd_ratings = np.sqrt(mean_squared_error(ratings_RS, ratings_LLM))

    # Print the results
    print("Mean Absolute Difference (MAD) for Ratings:", mad_ratings)
    print("Root Mean Squared Difference (RMSD) for Ratings:", rmsd_ratings)

    if not os.path.isfile(csv_file):
    # If the file doesn't exist, create it and write the header 
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['UserID', 'Genre', 'MovieInfo_RS', 'MovieInfo_LLM', 'Ratings_RS', 'Ratings_LLM', 'Rankings_RS','Rankings_LLM',
                            'RMSE_Ratings', 'MAE_Ratings', 'OverlapPercentage_Rankings',
                            'MAD_Ratings', 'RMSD_Ratings'])
        
        # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
            
    # Write the evaluation results for the current user and genre
        writer.writerow([user_id, desired_genres, movie_info_list_RS, movie_info_list_LLM, ratings_RS, ratings_LLM, rankings_RS, rankings_LLM,
                         rmse_ratings, mae_ratings, overlap_percentage_rankings, mad_ratings, rmsd_ratings])