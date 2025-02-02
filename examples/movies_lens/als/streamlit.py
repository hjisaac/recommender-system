import streamlit as st

## END DUPLICATION
# Sample movie data
movies_sample = [
    {
        "title": "Movie 1",
        "genres": "Action, Adventure",
        "rating": 4.2,
        "description": "An action-packed adventure full of thrills.",
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 2",
        "genres": "Comedy, Drama",
        "rating": 3.5,
        "description": "A heartwarming comedy with a touch of drama.",
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 2",
        "genres": "Comedy, Drama",
        "rating": 3.5,
        "description": "A heartwarming comedy with a touch of drama.",
        "image_url": "https://via.placeholder.com/150",
    },
    {
        "title": "Movie 2",
        "genres": "Comedy, Drama",
        "rating": 3.5,
        "description": "A heartwarming comedy with a touch of drama.",
        "image_url": "https://via.placeholder.com/150",
    },
    # Add more movies here...
]


# Function to render stars for ratings
def render_stars(rating):
    full_stars = int(rating)
    half_star = 1 if rating % 1 >= 0.5 else 0
    return "⭐" * full_stars + "✰" * half_star + "✰" * (5 - full_stars - half_star)


# Function to filter movies
def filter_movies(movies, search_term, genres):
    return [
        movie
        for movie in movies
        if search_term.lower() in movie["title"].lower()
        and (not genres or any(g.strip() in movie["genres"] for g in genres))
    ]


# Function to render the movie grid
def render_movie_grid(movies, columns_per_row):
    rows = (len(movies) + columns_per_row - 1) // columns_per_row
    for i in range(rows):
        cols = st.columns(columns_per_row)
        for j, col in enumerate(cols):
            index = i * columns_per_row + j
            if index < len(movies):
                movie = movies[index]
                with col:
                    st.markdown(
                        f"""
                        <div class="movie-container">
                            <h4>{movie["title"]}</h4>
                            <p><b>Genres:</b> {movie["genres"]}</p>
                            <p><b>Rating:</b> {render_stars(movie["rating"])} ({movie["rating"]}/5)</p>
                            <a href="?movie={index}">
                                <div class="zoom">
                                    <img src="{movie["image_url"]}" style="width: 100%;" alt="Movie Poster">
                                </div>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# Function to render movie details
def render_movie_details(movie):
    st.image(movie["image_url"], width=300)
    st.title(movie["title"])
    st.write(f"**Genres:** {movie['genres']}")
    st.write(f"**Rating:** {render_stars(movie['rating'])} ({movie['rating']}/5)")
    st.write(f"**Description:** {movie['description']}")
    if st.button("Go Back to Movie List"):
        st.query_params.clear()


# Streamlit Interface
st.title("Movie Grid 🎬")

# Check if the user clicked on a specific movie
query_params = st.query_params.to_dict()
movie_index = query_params.get("movie", [None])[0]

if movie_index is not None and movie_index.isdigit():
    # Display movie details if a movie is clicked
    movie_index = int(movie_index)
    if 0 <= movie_index < len(movies_sample):
        render_movie_details(movies_sample[movie_index])
    else:
        st.error("Invalid movie selected.")
else:
    # Filters
    search_term = st.text_input("Search movies...")
    genres = st.multiselect(
        "Filter by genres",
        options=list(
            set(
                g.strip() for movie in movies_sample for g in movie["genres"].split(",")
            )
        ),
    )

    # Filter movies based on input
    filtered_movies = filter_movies(movies_sample, search_term, genres)

    # Columns per row
    columns_per_row = st.slider("Columns per row", 1, 4, 3)

    # Add custom CSS for the zoom effect
    st.markdown(
        """
        <style>
            .zoom:hover {
                transform: scale(1.2);
                transition: transform 0.3s ease;
            }
            .movie-container {
                text-align: center;
                padding: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Show filtered movies or a "no results" message
    if not filtered_movies:
        st.warning("No movies found. Try adjusting your search or filters.")
    else:
        render_movie_grid(filtered_movies, columns_per_row)
