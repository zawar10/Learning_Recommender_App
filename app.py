from flask import Flask, request, jsonify, render_template
from recommender import (
    get_cf_recommendations,
    get_cb_recommendations,
    get_hybrid_recommendations,
    generate_learning_path,
    get_user_profile,
    get_all_users,
    get_all_courses,
    generate_custom_recommendations,
    generate_custom_learning_path
)

app = Flask(__name__)


# ─────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────

@app.route('/api/users', methods=['GET'])
def api_users():
    return jsonify({'users': get_all_users()})


@app.route('/api/courses', methods=['GET'])
def api_courses():
    return jsonify({'courses': get_all_courses()})


@app.route('/api/user/<int:user_id>', methods=['GET'])
def api_user_profile(user_id):
    profile = get_user_profile(user_id)
    if not profile:
        return jsonify({'error': f'User {user_id} not found'}), 404
    return jsonify(profile)


@app.route('/api/recommend/cf/<int:user_id>', methods=['GET'])
def api_cf(user_id):
    n    = int(request.args.get('n', 5))
    recs = get_cf_recommendations(user_id, n=n)
    return jsonify({'user_id': user_id, 'method': 'Collaborative Filtering', 'recommendations': recs})


@app.route('/api/recommend/cb', methods=['GET'])
def api_cb():
    course_ids = request.args.get('course_ids', '')
    n          = int(request.args.get('n', 5))
    try:
        ids = [int(x) for x in course_ids.split(',') if x]
    except ValueError:
        return jsonify({'error': 'Invalid course_ids. Use comma-separated integers.'}), 400
    recs = get_cb_recommendations(ids, n=n)
    return jsonify({'course_ids': ids, 'method': 'Content-Based Filtering', 'recommendations': recs})


@app.route('/api/recommend/hybrid/<int:user_id>', methods=['GET'])
def api_hybrid(user_id):
    n         = int(request.args.get('n', 5))
    cf_weight = float(request.args.get('cf_weight', 0.6))
    cb_weight = round(1 - cf_weight, 2)
    recs      = get_hybrid_recommendations(user_id, n=n, cf_weight=cf_weight, cb_weight=cb_weight)
    return jsonify({
        'user_id'  : user_id,
        'method'   : 'Hybrid',
        'cf_weight': cf_weight,
        'cb_weight': cb_weight,
        'recommendations': recs
    })


@app.route('/api/learning-path/<int:user_id>', methods=['GET'])
def api_learning_path(user_id):
    n_courses = int(request.args.get('n', 8))
    path, meta = generate_learning_path(user_id, n_courses=n_courses)
    if path is None:
        return jsonify({'error': f'Could not generate path for user {user_id}'}), 404
    return jsonify({'meta': meta, 'path': path})


@app.route('/api/recommend/custom', methods=['POST'])
def api_recommend_custom():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    completed_courses = data.get('completed_courses', [])
    skill_level = data.get('skill_level', 'Beginner')
    career_goal = data.get('career_goal', 'Data Analyst')
    n_recommendations = data.get('n_recommendations', 10)
    
    recommendations = generate_custom_recommendations(
        completed_courses, skill_level, career_goal, n_recommendations
    )
    
    learning_path = generate_custom_learning_path(recommendations)
    
    return jsonify({
        'career_goal': career_goal,
        'skill_level': skill_level,
        'recommended_courses': recommendations,
        'learning_path': learning_path
    })


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == '__main__':
    print("🚀 Starting Learning Path Recommender API...")
    print("   → Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
