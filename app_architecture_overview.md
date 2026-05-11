# Learning Recommender App: Architecture Overview

This document provides a comprehensive breakdown of your application, explaining how all the pieces fit together, what each component does, and the current state of the codebase.

## 🌟 The Big Picture
Your app is a **Personalized Course Recommendation Engine**. It acts like a smart advisor for learners, looking at their past behavior (ratings) and profile data (career goals, skill level, available time) to suggest the best courses for them to take next. It can also generate a structured, week-by-week learning path.

---

## 🛠️ Technology Stack
- **Backend Framework**: Python with **Flask** (serves the API and the web page).
- **Machine Learning / Logic**: Python with `pandas`, `numpy`, and `scikit-learn` (handles data manipulation, matrix operations, and scoring).
- **Frontend**: Standard HTML, Vanilla CSS, and Vanilla JavaScript (No heavy frameworks like React, keeping it fast and simple).
- **Data Persistence**: Pre-trained machine learning artifacts and CSV files stored in a `model/` directory (loaded via `joblib`).

---

## 🧠 The Core Engine (`recommender.py`)
This is the "brain" of your application. When the Flask server starts, this file loads several machine learning models and datasets into memory:
- **Datasets**: `courses.csv`, `users.csv`, `ratings.csv`.
- **Pre-computed Matrices**: `R_predicted.pkl` (for collaborative filtering), `cosine_sim.pkl` (for content-based similarity).

It exposes several powerful functions:

1. **Collaborative Filtering (`get_cf_recommendations`)**: Suggests courses by predicting what rating a user *would* give to unseen courses, based on the `R_predicted` matrix. (i.e., "Users who liked what you liked also liked these").
2. **Content-Based Filtering (`get_cb_recommendations`)**: Suggests courses based on the attributes of courses a user has already taken (using `cosine_sim`).
3. **Hybrid Filtering (`get_hybrid_recommendations`)**: Combines both Collaborative and Content-Based models to provide a more robust, balanced recommendation.
4. **Learning Path Generator (`generate_learning_path`)**: This is the most advanced feature. It takes the Hybrid recommendations, filters them to match the user's career goal, orders them logically by difficulty (Beginner → Intermediate → Advanced), and chunks them into a week-by-week syllabus based on the user's available study hours.
5. **User Profiling (`get_user_profile`)**: Retrieves a user's stats (courses taken, average rating) and **their complete learning history**, which we recently added.

---

## 🔌 The API Layer (`app.py`)
This file connects the "brain" to the internet. It sets up a Flask web server running on `http://127.0.0.1:5000`.

- **Frontend Route**:
  - `/` → Serves the `index.html` file.
- **REST API Routes** (These return JSON data):
  - `/api/users` → Returns a list of all learners.
  - `/api/courses` → Returns a list of all courses in the catalog.
  - `/api/user/<user_id>` → Returns a specific user's profile and history.
  - `/api/recommend/cf/<user_id>` → Returns Collaborative Filtering recommendations.
  - `/api/recommend/cb` → Returns Content-Based recommendations.
  - `/api/recommend/hybrid/<user_id>` → Returns Hybrid recommendations.
  - `/api/learning-path/<user_id>` → Returns the generated learning path.

---

## 💻 The Frontend (`templates/index.html`)
This is what the user sees in their browser. It's a clean, single-page application with a modern gradient design. 

**User Flow & Features**:
1. **User Selection**: The page fetches all users from `/api/users` and populates a dropdown.
2. **Tabbed Interface**: Once a user is selected, two tabs appear:
   - **Recommendations**: Lets the user choose between "Based on my history" (Collaborative Filtering) or "My Learning Path". Clicking the "Show Me" button hits the respective API endpoint and renders the courses as visually appealing cards.
   - **Learner History**: (The feature we just built!) Automatically hits `/api/user/<id>` to display the user's profile stats (total courses, average rating) and a list of every course they've ever rated, complete with difficulty tags and star ratings.

## 🔄 Data Flow Example (Requesting a Learning Path)
1. You select "Learner 1" and click "My Learning Path" in the browser.
2. The Javascript in `index.html` sends a `GET` request to `/api/learning-path/1?n=8`.
3. `app.py` catches this request and calls `generate_learning_path(1, n_courses=8)` in `recommender.py`.
4. `recommender.py` looks up Learner 1, gets 20 Hybrid recommendations, filters them by Learner 1's career goal, sorts them by difficulty, calculates how many weeks it will take based on Learner 1's available hours, and returns a Python dictionary.
5. `app.py` converts that dictionary to JSON and sends it back to the browser.
6. The Javascript parses the JSON and renders the course cards dynamically on the screen.
