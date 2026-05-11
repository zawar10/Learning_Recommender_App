import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────

R_predicted    = joblib.load('model/R_predicted.pkl')
user_idx       = joblib.load('model/user_idx.pkl')
course_idx     = joblib.load('model/course_idx.pkl')
cosine_sim     = joblib.load('model/cosine_sim.pkl')
scaler         = joblib.load('model/scaler.pkl')
course_indices = joblib.load('model/course_indices.pkl')

courses    = pd.read_csv('model/courses.csv')
users      = pd.read_csv('model/users.csv')
ratings_df = pd.read_csv('model/ratings.csv')

with open('model/goal_category_map.json') as f:
    goal_category_map = json.load(f)

difficulty_order = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}

course_popularity = {}
if not ratings_df.empty:
    avg_ratings = ratings_df.groupby('course_id')['rating'].mean()
    max_r = avg_ratings.max()
    min_r = avg_ratings.min()
    if max_r > min_r:
        course_popularity = ((avg_ratings - min_r) / (max_r - min_r)).to_dict()
    else:
        course_popularity = {cid: 0.5 for cid in avg_ratings.index}


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def predict_rating(user_id, course_id):
    if user_id not in user_idx or course_id not in course_idx:
        return 3.0
    return float(np.clip(
        R_predicted[user_idx[user_id], course_idx[course_id]], 1, 5
    ))


def get_cf_recommendations(user_id, n=10):
    rated      = set(ratings_df[ratings_df['user_id'] == user_id]['course_id'])
    candidates = list(set(courses['course_id']) - rated)
    scored     = sorted([(c, predict_rating(user_id, c)) for c in candidates],
                        key=lambda x: x[1], reverse=True)[:n]
    result = []
    for course_id, pred in scored:
        info = courses[courses['course_id'] == course_id].iloc[0]
        result.append({
            'course_id'  : int(course_id),
            'course_name': info['course_name'],
            'category'   : info['category'],
            'difficulty' : info['difficulty'],
            'pred_rating': round(pred, 2)
        })
    return result


def get_cb_recommendations(course_ids, n=10):
    if isinstance(course_ids, int):
        course_ids = [course_ids]
    sim_scores = np.zeros(len(courses))
    for cid in course_ids:
        if cid in course_indices.index:
            sim_scores += cosine_sim[course_indices[cid]]
    sim_scores /= max(len(course_ids), 1)

    sim_df = pd.DataFrame({'course_id': courses['course_id'], 'similarity': sim_scores})
    sim_df = sim_df[~sim_df['course_id'].isin(course_ids)]
    sim_df = sim_df.sort_values('similarity', ascending=False).head(n)
    result = sim_df.merge(courses[['course_id', 'course_name', 'category', 'difficulty']], on='course_id')
    return result.to_dict(orient='records')


def get_hybrid_recommendations(user_id, n=10, cf_weight=0.6, cb_weight=0.4):
    rated      = set(ratings_df[ratings_df['user_id'] == user_id]['course_id'])
    candidates = list(set(courses['course_id']) - rated)

    cf_scores = {cid: predict_rating(user_id, cid) for cid in candidates}

    liked = ratings_df[
        (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)
    ]['course_id'].tolist()

    if liked:
        sim_scores = np.zeros(len(courses))
        for cid in liked:
            if cid in course_indices.index:
                sim_scores += cosine_sim[course_indices[cid]]
        sim_scores /= len(liked)
        cb_scores = {int(courses.iloc[i]['course_id']): sim_scores[i] for i in range(len(courses))}
    else:
        cb_scores = {cid: 0.5 for cid in candidates}

    cf_vals = np.array([cf_scores[c] for c in candidates]).reshape(-1, 1)
    cb_vals = np.array([cb_scores.get(c, 0) for c in candidates]).reshape(-1, 1)

    sc = MinMaxScaler()
    cf_norm = sc.fit_transform(cf_vals).flatten()
    sc2 = MinMaxScaler()
    cb_norm = sc2.fit_transform(cb_vals).flatten()

    hybrid_scores = cf_weight * cf_norm + cb_weight * cb_norm
    scored = sorted(zip(candidates, hybrid_scores), key=lambda x: x[1], reverse=True)[:n]

    result = []
    for course_id, score in scored:
        info = courses[courses['course_id'] == course_id].iloc[0]
        result.append({
            'course_id'   : int(course_id),
            'course_name' : info['course_name'],
            'category'    : info['category'],
            'difficulty'  : info['difficulty'],
            'hybrid_score': round(float(score), 4)
        })
    return result


def generate_learning_path(user_id, n_courses=8):
    if user_id not in users['user_id'].values:
        return None, None

    user      = users[users['user_id'] == user_id].iloc[0]
    goal      = user['career_goal']
    skill_lvl = user['skill_level']
    hours_wk  = int(user['available_hours_per_week'])

    recs = pd.DataFrame(get_hybrid_recommendations(user_id, n=20))
    if recs.empty:
        return None, None

    recs = recs.merge(courses[['course_id', 'duration_hours']], on='course_id')

    relevant_cats = goal_category_map.get(goal, [])
    filtered = recs[recs['category'].isin(relevant_cats)]
    if len(filtered) < n_courses:
        filtered = recs

    level_priority_map = {
        'Beginner'    : {0: 0, 1: 1, 2: 2},
        'Intermediate': {1: 0, 0: 1, 2: 2},
        'Advanced'    : {2: 0, 1: 1, 0: 2},
    }
    filtered = filtered.copy()
    filtered['diff_order']    = filtered['difficulty'].map(difficulty_order)
    filtered['level_priority'] = filtered['diff_order'].map(
        level_priority_map.get(skill_lvl, {0: 0, 1: 1, 2: 2})
    )
    filtered = filtered.sort_values(['level_priority', 'hybrid_score'], ascending=[True, False])
    path = filtered.head(n_courses).reset_index(drop=True)

    total_hours  = int(path['duration_hours'].sum())
    weeks_needed = int(np.ceil(total_hours / hours_wk))

    path_list = []
    cumulative = 0
    for i, row in path.iterrows():
        week_start  = int(np.floor(cumulative / hours_wk)) + 1
        cumulative += row['duration_hours']
        week_end    = int(np.floor(cumulative / hours_wk)) + 1
        path_list.append({
            'step'        : i + 1,
            'course_id'   : int(row['course_id']),
            'course_name' : row['course_name'],
            'category'    : row['category'],
            'difficulty'  : row['difficulty'],
            'duration_hrs': int(row['duration_hours']),
            'week_start'  : week_start,
            'week_end'    : week_end,
        })

    meta = {
        'user_id'      : int(user_id),
        'career_goal'  : goal,
        'skill_level'  : skill_lvl,
        'hours_per_week': hours_wk,
        'total_hours'  : total_hours,
        'weeks_needed' : weeks_needed,
    }
    return path_list, meta


def get_user_profile(user_id):
    if user_id not in users['user_id'].values:
        return None
    user   = users[users['user_id'] == user_id].iloc[0].to_dict()
    user['username'] = f"Learner {user_id}"  # Add username
    rated  = ratings_df[ratings_df['user_id'] == user_id]
    rated  = rated.merge(courses[['course_id', 'course_name', 'difficulty']], on='course_id')
    user['rated_courses'] = len(rated)  # Just the count
    user['total_rated']   = len(rated)
    user['avg_rating']    = float(rated['rating'].mean()) if len(rated) > 0 else 0.0
    user['history']       = rated[['course_id', 'course_name', 'difficulty', 'rating']].to_dict(orient='records')
    return user


def get_all_users():
    result = []
    for _, row in users.iterrows():
        result.append({
            'user_id': int(row['user_id']),
            'username': f"Learner {row['user_id']}"  # Create a friendly name
        })
    return result


def get_all_courses():
    result = []
    cat_to_goals = {}
    for goal, cats in goal_category_map.items():
        for cat in cats:
            if cat not in cat_to_goals:
                cat_to_goals[cat] = []
            cat_to_goals[cat].append(goal)

    for _, row in courses.iterrows():
        desc = row.get('description')
        desc_str = '' if pd.isna(desc) else str(desc)
        c_dict = {
            'course_id': int(row['course_id']),
            'course_name': row['course_name'],
            'category': row['category'],
            'difficulty': row['difficulty'],
            'duration_hours': int(row['duration_hours']),
            'description': desc_str,
        }
        features = str(row.get('features', ''))
        tags = list(set([t.lower() for t in features.split() if len(t) > 3]))
        career_paths = cat_to_goals.get(row['category'], [])
        
        c_dict['tags'] = tags[:5]
        c_dict['career_paths'] = career_paths
        result.append(c_dict)
    return result


def generate_custom_recommendations(completed_courses, skill_level, career_goal, n_recommendations=10):
    candidates = list(set(courses['course_id']) - set(completed_courses))
    
    sim_scores = np.zeros(len(courses))
    valid_completed = [cid for cid in completed_courses if cid in course_indices.index]
    if valid_completed:
        for cid in valid_completed:
            sim_scores += cosine_sim[course_indices[cid]]
        sim_scores /= len(valid_completed)
    cb_scores = {int(courses.iloc[i]['course_id']): sim_scores[i] for i in range(len(courses))}
    
    if cb_scores:
        max_cb = max(cb_scores.values()) if max(cb_scores.values()) > 0 else 1.0
        cb_scores = {k: v/max_cb for k, v in cb_scores.items()}

    relevant_cats = goal_category_map.get(career_goal, [])
    
    scored = []
    for cid in candidates:
        info = courses[courses['course_id'] == cid].iloc[0]
        
        content_sim = cb_scores.get(cid, 0)
        career_match = 1.0 if info['category'] in relevant_cats else 0.0
        popularity = course_popularity.get(cid, 0.5)
        
        final_score = 0.5 * content_sim + 0.3 * career_match + 0.2 * popularity
        
        if info['difficulty'] == skill_level:
            final_score += 0.1
            
        scored.append((cid, final_score))
        
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    result = []
    for course_id, score in scored:
        info = courses[courses['course_id'] == course_id].iloc[0]
        d = info.get('description')
        desc_str = '' if pd.isna(d) else str(d)
        result.append({
            'course_id'   : int(course_id),
            'course_name' : info['course_name'],
            'category'    : info['category'],
            'difficulty'  : info['difficulty'],
            'duration_hrs': int(info['duration_hours']),
            'relevance_score': round(float(score), 4),
            'description': desc_str,
            'tags': list(set([t.lower() for t in str(info.get('features', '')).split() if len(t) > 3]))[:5]
        })
    return result


def generate_custom_learning_path(recommendations, hours_wk=10):
    if not recommendations:
        return []
        
    df = pd.DataFrame(recommendations)
    df['diff_order'] = df['difficulty'].map(difficulty_order)
    df = df.sort_values(['diff_order', 'relevance_score'], ascending=[True, False])
    
    path_list = []
    cumulative = 0
    for i, row in df.iterrows():
        week_start  = int(np.floor(cumulative / hours_wk)) + 1
        cumulative += row['duration_hrs']
        week_end    = int(np.floor(cumulative / hours_wk)) + 1
        path_desc = row.get('description', np.nan)
        path_desc_str = '' if pd.isna(path_desc) else str(path_desc)
        path_list.append({
            'step'        : len(path_list) + 1,
            'course_id'   : int(row['course_id']),
            'course_name' : row['course_name'],
            'category'    : row['category'],
            'difficulty'  : row['difficulty'],
            'duration_hrs': int(row['duration_hrs']),
            'week_start'  : week_start,
            'week_end'    : week_end,
            'relevance_score': row['relevance_score'],
            'description': path_desc_str,
            'tags': row['tags']
        })
    return path_list
