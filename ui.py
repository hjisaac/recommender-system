import streamlit as st

# Sample movie data
movies_sample = [
    {
        "title": "Movie 1",
        "genres": "Action, Adventure",
        "rating": 4.2,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 2",
        "genres": "Comedy, Drama",
        "rating": 3.5,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 3",
        "genres": "Romance, Drama",
        "rating": 4.8,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 4",
        "genres": "Horror, Mystery",
        "rating": 2.9,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 5",
        "genres": "Sci-Fi, Thriller",
        "rating": 4.0,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 6",
        "genres": "Fantasy, Family",
        "rating": 3.8,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 7",
        "genres": "Action, Drama",
        "rating": 5.0,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 8",
        "genres": "Animation, Family",
        "rating": 4.1,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 9",
        "genres": "Documentary, History",
        "rating": 3.7,
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 10",
        "genres": "Adventure, Fantasy",
        "rating": 4.5,
        "image_url": "https://via.placeholder.com/150",
    },
]

# Streamlit Interface
st.title("Movie Grid 🎬")

# Search bar
search_term = st.text_input("Search movies...")

# Filter movies based on search term
filtered_movies = [
    movie for movie in movies_sample if search_term.lower() in movie["title"].lower()
]

# Define the number of columns in the grid
columns_per_row = st.slider("Columns per row", 1, 4, 3)  # Allow user to adjust columns

# Calculate rows based on columns
rows = (len(filtered_movies) + columns_per_row - 1) // columns_per_row

# Add some custom CSS for the zoom-in effect
st.markdown(
    """
    <style>
        .zoom:hover {
            transform: scale(1.2);
            transition: transform 0.3s ease;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Loop through movies and display in a grid
for i in range(rows):
    cols = st.columns(columns_per_row)
    for j, col in enumerate(cols):
        index = i * columns_per_row + j
        if index < len(filtered_movies):
            movie = filtered_movies[index]
            with col:
                # Create an empty container to align text and image
                movie_info = f"### {movie['title']}\n**Genres:** {movie['genres']}\n**Rating:** {movie['rating']} ⭐"
                st.markdown(movie_info)

                # Display movie image with zoom effect
                st.markdown(
                    f'<div class="zoom"><img src="{movie["image_url"]}" style="width: 100%;" alt="Movie Poster"></div>',
                    unsafe_allow_html=True,
                )
                st.write("\n\n")

# Add some spacing at the bottom
st.write("")
