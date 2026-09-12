import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from your .env file."
    )

# OpenRouter client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Flask
app = Flask(__name__)
CORS(app)


def create_prompt(resume, role):
    return f"""
You are a professional ATS Resume Analyzer and HR Interview Evaluator.

Analyze this resume for the job role: {role}

RESUME:
{resume}

Give the answer in this format:

1. Resume Summary Quality (0-10)

2. Job Role Match Score (0-10)

3. Key Strengths Observed

4. Weak Areas and Points to Improve

5. Suggested Projects to Add

6. Fully Improved Resume Version
Rewrite professionally but DO NOT invent any information.

7. Top 10 Interview Questions to Prepare

Keep the analysis practical and specific to the job role.
"""


@app.route("/analyze", methods=["POST"])
def analyze_resume():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "detail": "No JSON data received."
            }), 400

        resume = data.get("resume", "").strip()
        role = data.get("role", "").strip()

        if not resume:
            return jsonify({
                "detail": "Resume content is required."
            }), 400

        if not role:
            return jsonify({
                "detail": "Job role is required."
            }), 400

        print("Sending request to OpenRouter...")

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional ATS Resume Analyzer."
                },
                {
                    "role": "user",
                    "content": create_prompt(resume, role)
                }
            ],
            temperature=0.2
        )

        result = response.choices[0].message.content

        if not result:
            return jsonify({
                "detail": "OpenRouter returned an empty response."
            }), 500

        print("Analysis completed successfully.")

        return jsonify({
            "result": result
        }), 200

    except Exception as e:

        print("\n========== BACKEND ERROR ==========")
        print(repr(e))
        print("===================================\n")

        return jsonify({
            "detail": f"OpenRouter error: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("Starting Resume Analyzer backend...")
    print("Backend URL: http://localhost:8000")

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )