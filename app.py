from flask import Flask, request, jsonify
import psycopg2
import joblib
from dotenv import load_dotenv
import os

app = Flask(__name__)

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

model_paths = {
    'software_engineer': 'models/se_model.pkl',
    'analyst': 'models/an_model.pkl'
}

models = {key: joblib.load(path) for key, path in model_paths.items()}

@app.route('/get_top_resumes', methods=['POST'])
def get_top_resumes():
    data = request.json
    vacancy_type = data['vacancy_type']
    resume_ids = data['resume_ids']
    top_x = data['top_x']

    with conn.cursor() as cursor:
        cursor.execute("SELECT id, job_name, skills FROM resume WHERE id = ANY(%s)", (resume_ids,))
        resumes = cursor.fetchall()

    model = models.get(vacancy_type)

    resume_features = [{'id': resume[0], 'job_name': resume[1], 'skills': resume[2]} for resume in resumes]

    # resume_scores = [(resume['id'], model.predict([resume['skills']])[0]) for resume in resume_features]
    resume_scores = [(resume['id'], 50) for resume in resume_features]

    sorted_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)

    top_resumes = sorted_resumes[:top_x]
    top_resume_ids = [resume[0] for resume in top_resumes]

    return jsonify(top_resume_ids)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
