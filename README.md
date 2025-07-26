# Movie Sentiment & Recommendation System

This project is a web application that recommends movies to users based on the sentiment analysis of their input text. Users can describe their mood or what they feel like watching, and the system provides a list of suitable movie recommendations.

## 🚀 Features

- **Sentiment Analysis:** Analyzes the user's text input to determine the underlying sentiment (e.g., positive, negative, neutral).
- **Movie Recommendation:** Suggests movies from a dataset that match the user's sentiment.
- **Web Interface:** A simple and intuitive frontend for users to interact with the system.
- **Containerized:** The entire application is containerized using Docker for easy setup and deployment.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI
- **Machine Learning:** NLTK (for sentiment analysis), Scikit-learn (for content-based filtering)
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker, Docker Compose

## 📦 Setup and Installation

To get the project up and running locally, follow these simple steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Ensure you have Docker and Docker Compose installed** on your machine.

3.  **Run the application:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the application:**
    Open your web browser and navigate to `http://localhost:80`.

## Usage

1.  Once the application is running, you will see a text input field.
2.  Enter a sentence or a paragraph describing your current mood or the type of movie you want to watch (e.g., "I want to watch a happy and uplifting movie" or "I'm in the mood for something dark and thrilling").
3.  Click the "Get Recommendations" button.
4.  The system will analyze your text and display a list of recommended movies that match your sentiment.

