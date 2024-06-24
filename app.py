from flask import Flask, request, jsonify
import psycopg2
import joblib
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

app = Flask(__name__)

model_paths = {
    'Аналитик': 'models/analyst_model.pkl'
    # for now
}

models = {key: joblib.load(path) for key, path in model_paths.items()}


conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

education_levels = {
    "Среднее": 1,
    "Среднее специальное": 2,
    "Неоконченное высшее": 3,
    "Высшее": 4,
    "Бакалавр": 5,
    "Магистр": 6,
    "Кандидат наук": 7,
    "Доктор наук": 8
}

analyst_skills_list = ['python', 'sql', 'english', 'git', 'linux', 'spark', 'hadoop', 'machine_learning', 'bi', 'pandas']

def fetch_resumes(resume_ids):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, job_name, education, skills FROM resume WHERE id = ANY(%s)", (resume_ids,))
        return cursor.fetchall()

def fetch_experience(resume_ids):
    with conn.cursor() as cursor:
        cursor.execute("SELECT resume_id, time_from, time_to FROM experience WHERE resume_id = ANY(%s)", (resume_ids,))
        return cursor.fetchall()

def calculate_experience(experience_data):
    experience_dict = {}
    for exp in experience_data:
        resume_id, time_from, time_to = exp
        time_from = time_from if time_from else datetime.now()
        time_to = time_to if time_to else datetime.now()
        months_of_experience = (time_to.year - time_from.year) * 12 + time_to.month - time_from.month
        if resume_id in experience_dict:
            experience_dict[resume_id] += months_of_experience
        else:
            experience_dict[resume_id] = months_of_experience
    return experience_dict


def generate_features(resume_data, experience_data, vacancy_type):
    experience_dict = calculate_experience(experience_data)
    features = []

    for resume in resume_data:
        resume_id, job_name, education, skills = resume

        position_feature = 1 if job_name == vacancy_type else 0
        education_feature = education_levels.get(education, 0)
        experience_feature = experience_dict.get(resume_id, 0)
        skill_features = [1 if skill in skills else 0 for skill in analyst_skills_list]

        feature_list = {
            'position': position_feature,
            'education': education_feature,
            'experience': experience_feature,
        }
        feature_list.update({skill: skill_feature for skill, skill_feature in zip(analyst_skills_list, skill_features)})
        features.append((resume_id, feature_list))

    return features

def predict(loaded_trees, X_new):
    y_pred = 0.6708860759493671  
    for tree in loaded_trees:
        y_pred += tree.predict(X_new)
    return sigmoid(y_pred)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

@app.route('/get_top_resumes', methods=['POST'])
def get_top_resumes():
    data = request.json
    vacancy_type = data['vacancy_type']
    resume_ids = data['resume_ids']
    top_x = 2

    analyst_features = {
        'position': 0, 'education': 0, 'experience':0, 'python': 0, 'sql': 0, 'english': 0, 'git': 0, 'linux': 0, 
        'spark': 0, 'hadoop': 0, 'machine_learning': 0, 'bi': 0, 'pandas': 0
    }

    resume_data = fetch_resumes(resume_ids)
    print(resume_data)
    experience_data = fetch_experience(resume_ids)
    print(experience_data)
    features = generate_features(resume_data, experience_data, vacancy_type)
    print(features)

    feature_list = [f[1] for f in features]
    model = models.get('vacancy_type')
    scores = predict(model, feature_list)
    resume_scores = [(features[i][0], scores[i]) for i in range(len(scores))]
    sorted_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)
    sorted_resume_ids = [resume[0] for resume in sorted_resumes]

    return jsonify(sorted_resume_ids[:top_x])

if __name__ == '__main__':
   app.run(host='127.0.0.1', port=5000)
