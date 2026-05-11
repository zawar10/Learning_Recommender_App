# Learning Recommender App

A personalized course recommendation engine that helps learners discover the best courses for their goals and skill level. Using collaborative filtering, content-based filtering, and hybrid recommendation algorithms, this app generates intelligent course suggestions and structured learning paths tailored to each user's profile.

## Features

- **🤖 Multiple Recommendation Engines**
  - **Collaborative Filtering**: Recommends courses based on similar users' preferences ("Users who liked what you liked also liked...")
  - **Content-Based Filtering**: Suggests courses similar to ones you've already taken
  - **Hybrid Approach**: Combines both methods for more robust, balanced recommendations

- **📚 Intelligent Learning Paths**
  - Generates week-by-week structured learning plans
  - Sorts courses by difficulty progression (Beginner → Intermediate → Advanced)
  - Adapts to your career goals and available study hours
  - Shows estimated completion time

- **👤 Learner Profiles**
  - View complete learning history with ratings
  - Track courses taken and average ratings
  - See profile stats and career goals

- **🎯 Custom Recommendations**
  - Get personalized suggestions based on your skill level and career goals
  - Exclude already completed courses automatically

## Tech Stack

- **Backend**: Python with Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: Vanilla HTML, CSS, and JavaScript
- **Data**: Pre-trained ML models (joblib) and CSV datasets

## Project Structure

```
learning_recommender_app/
├── app.py                           # Flask application and API routes
├── recommender.py                   # Core recommendation engine
├── requirements.txt                 # Python dependencies
├── app_architecture_overview.md    # Detailed architecture documentation
├── model/
│   ├── courses.csv                 # Course catalog
│   ├── users.csv                   # User profiles
│   ├── ratings.csv                 # User ratings data
│   ├── R_predicted.pkl             # Pre-trained collaborative filtering matrix
│   ├── cosine_sim.pkl              # Content-based similarity matrix
│   ├── user_idx.pkl                # User index mapping
│   ├── course_idx.pkl              # Course index mapping
│   ├── course_indices.pkl          # Course indices for similarity lookup
│   ├── scaler.pkl                  # Normalized scaler for ratings
│   └── goal_category_map.json      # Career goal to course category mapping
└── templates/
    └── index.html                  # Single-page frontend application
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository** (or navigate to the project folder)
   ```bash
   cd learning_recommender_app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Make sure your virtual environment is activated**

2. **Run the Flask server**
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

### Using the Dashboard

1. **Select a User**: Choose from the dropdown to view recommendations for that learner
2. **View Recommendations**: Choose between:
   - "Based on my history" for collaborative filtering recommendations
   - "My Learning Path" for a structured week-by-week course plan
3. **Check Learner History**: View the user's complete profile, stats, and all courses they've rated
4. **Explore Courses**: Browse all available courses in the catalog

## API Endpoints

### User Information
- `GET /api/users` - Get all users
- `GET /api/courses` - Get all courses
- `GET /api/user/<user_id>` - Get specific user's profile and history

### Recommendations
- `GET /api/recommend/cf/<user_id>?n=5` - Collaborative filtering recommendations
  - `n`: Number of recommendations (default: 5)
  
- `GET /api/recommend/cb?course_ids=1,2,3&n=5` - Content-based recommendations
  - `course_ids`: Comma-separated course IDs to base recommendations on
  - `n`: Number of recommendations (default: 5)
  
- `GET /api/recommend/hybrid/<user_id>?n=5&cf_weight=0.6` - Hybrid recommendations
  - `n`: Number of recommendations (default: 5)
  - `cf_weight`: Weight for collaborative filtering (0-1, default: 0.6)
  
- `POST /api/recommend/custom` - Custom recommendations based on criteria
  ```json
  {
    "completed_courses": [1, 2, 3],
    "skill_level": "Intermediate",
    "career_goal": "Data Scientist",
    "n_recommendations": 10
  }
  ```

### Learning Paths
- `GET /api/learning-path/<user_id>?n=8` - Generate a learning path
  - `n`: Number of courses to include (default: 8)

## How It Works

### Data Flow

1. **Request**: User selects a learner and requests recommendations in the web interface
2. **API Call**: JavaScript sends a request to the appropriate API endpoint
3. **Processing**: 
   - The Flask app receives the request and calls the recommender engine
   - The recommender loads pre-trained ML models and datasets
   - Algorithms score candidate courses based on the selected method
4. **Response**: JSON data is sent back to the frontend
5. **Display**: JavaScript renders the recommendations as interactive cards

### Recommendation Algorithms

**Collaborative Filtering (CF)**
- Uses a pre-computed predicted rating matrix
- Predicts how much a user would rate unseen courses
- Returns top-scored courses the user hasn't rated yet

**Content-Based (CB)**
- Computes similarity between course attributes
- Based on courses the user has already rated
- Recommends similar courses they might enjoy

**Hybrid**
- Combines CF and CB scores with configurable weights
- CF Weight: 60% (default), CB Weight: 40%
- Provides balanced, robust recommendations

**Learning Path Generator**
- Gets 20 hybrid recommendations
- Filters by user's career goal category
- Sorts by difficulty level
- Chunks into weekly schedule based on available study hours
- Returns structured week-by-week plan

## Model Artifacts

The app uses pre-trained machine learning models stored in the `model/` directory:

| File | Purpose |
|------|---------|
| `R_predicted.pkl` | Collaborative filtering prediction matrix |
| `cosine_sim.pkl` | Content-based course similarity matrix |
| `user_idx.pkl` | Mapping of user IDs to matrix indices |
| `course_idx.pkl` | Mapping of course IDs to matrix indices |
| `course_indices.pkl` | Course indices for similarity lookup |
| `scaler.pkl` | MinMax scaler for normalized ratings |

## Data Files

| File | Description |
|------|-------------|
| `courses.csv` | Course catalog with names, categories, difficulty levels |
| `users.csv` | User profiles with career goals and skill levels |
| `ratings.csv` | User-course ratings (1-5 scale) |
| `goal_category_map.json` | Maps career goals to relevant course categories |

## Configuration

Key settings can be adjusted in `recommender.py`:

- **Difficulty Order**: Beginner → Intermediate → Advanced
- **Similarity Threshold**: Minimum similarity score for content-based recommendations
- **Learning Path Duration**: Weeks and hours per week for path generation

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify the Flask run command:
```bash
python -c "from app import app; app.run(port=5001)"
```

### Missing Model Files
Ensure all `.pkl` files are present in the `model/` directory. If training models from scratch is needed, refer to the model training scripts (not included in this repo).

### No Recommendations
- Check that ratings exist in `ratings.csv`
- Verify user ID exists in `users.csv`
- Ensure courses exist in `courses.csv`

## Future Enhancements

- [ ] User authentication and personalized dashboards
- [ ] Real-time model updates from new rating data
- [ ] Advanced filtering (by category, difficulty, rating)
- [ ] Course completion tracking
- [ ] Adaptive learning paths based on progress
- [ ] API documentation with Swagger/OpenAPI
- [ ] Docker deployment configuration

## Development

### Adding New Features

1. **New Recommendation Algorithm**: Add function in `recommender.py`, expose via endpoint in `app.py`
2. **Frontend Changes**: Modify `templates/index.html`
3. **New Data Fields**: Update CSV files and retrain models as needed

### Testing

Manual testing via the web interface or API endpoints with tools like `curl` or Postman.

## License

[Add your license here]

## Contact

[Add contact information here]

---

**Enjoy personalized learning recommendations!** 🚀
