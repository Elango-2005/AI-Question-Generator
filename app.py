import os
import json
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Question Generator")


# ============================================================
# TEMPORARY TEST STORAGE
# ============================================================

tests = {}


# ============================================================
# QUESTION TYPES
# ============================================================

question_types = {

    "MCQs": """
Generate exactly 5 multiple-choice questions.

Requirements:
- Exactly 5 questions.
- Exactly 4 options per question.
- Options must be A, B, C and D.
- Questions must be based only on the provided study content.
- Difficulty should range from beginner to intermediate.
- Include conceptual, practical and code-based questions where appropriate.
- Provide the correct option letter.
- Provide a clear explanation.
""",

    "Short Answer": """
Generate exactly 5 short-answer questions.

Requirements:
- Exactly 5 questions.
- Questions must be based only on the provided study content.
- Focus on important concepts.
- Difficulty should range from beginner to intermediate.
- Provide a concise model answer.
- Provide a useful explanation.
""",

    "Interview": """
Generate exactly 5 technical interview questions.

Requirements:
- Exactly 5 questions.
- Questions must be based only on the provided study content.
- Include conceptual, practical and code-based questions.
- Suitable for a beginner-level technical interview.
- Provide a concise interview-quality model answer.
- Provide 2 to 4 important points that a candidate should mention.
- Provide useful additional context.
"""
}


# ============================================================
# GEMINI QUESTION GENERATOR
# ============================================================

def question_generator(content, question_type):

    system_prompt = f"""
You are an AI Question Generator.

Generate questions based ONLY on the study content provided by the user.

Question Type:
{question_type}

{question_types[question_type]}

IMPORTANT:
Return ONLY valid JSON.

Do NOT return:
- Markdown
- ```json
- ```
- Any text before the JSON
- Any text after the JSON

The JSON must follow this exact structure:

{{
    "question_type": "{question_type}",
    "questions": [
        {{
            "number": 1,
            "question": "Question text",
            "options": [],
            "correct_answer": "",
            "explanation": "",
            "model_answer": "",
            "evaluation_points": []
        }}
    ]
}}

MCQ rules:

- options must contain exactly 4 objects.
- Every option object MUST contain:
    "letter"
    "text"
- letter must be exactly one of:
    A
    B
    C
    D
- text must contain the complete option text.
- correct_answer must contain ONLY A, B, C or D.
- explanation must explain the correct answer.
- model_answer must be an empty string.
- evaluation_points must be an empty array.

Short Answer rules:

- options must be an empty array.
- correct_answer must be an empty string.
- model_answer must contain the ideal answer.
- explanation must contain useful clarification.
- evaluation_points must be an empty array.

Interview rules:

- options must be an empty array.
- correct_answer must be an empty string.
- model_answer must contain an interview-quality answer.
- explanation must contain additional context.
- evaluation_points must contain 2 to 4 important points.

Never use undefined, null or missing values.

Make sure all strings are valid JSON strings.
"""


    # ========================================================
    # STRICT JSON SCHEMA
    # ========================================================

    option_schema = types.Schema(
        type="OBJECT",
        properties={
            "letter": types.Schema(
                type="STRING"
            ),
            "text": types.Schema(
                type="STRING"
            )
        },
        required=[
            "letter",
            "text"
        ]
    )


    question_schema = types.Schema(
        type="OBJECT",
        properties={

            "number": types.Schema(
                type="INTEGER"
            ),

            "question": types.Schema(
                type="STRING"
            ),

            "options": types.Schema(
                type="ARRAY",
                items=option_schema
            ),

            "correct_answer": types.Schema(
                type="STRING"
            ),

            "explanation": types.Schema(
                type="STRING"
            ),

            "model_answer": types.Schema(
                type="STRING"
            ),

            "evaluation_points": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="STRING"
                )
            )
        },

        required=[
            "number",
            "question",
            "options",
            "correct_answer",
            "explanation",
            "model_answer",
            "evaluation_points"
        ]
    )


    response_schema = types.Schema(
        type="OBJECT",

        properties={

            "question_type": types.Schema(
                type="STRING"
            ),

            "questions": types.Schema(
                type="ARRAY",
                items=question_schema
            )
        },

        required=[
            "question_type",
            "questions"
        ]
    )


    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    response = client.models.generate_content(

        model="gemini-3.5-flash-lite",

        config=types.GenerateContentConfig(

            system_instruction=system_prompt,

            max_output_tokens=4000,

            response_mime_type="application/json",

            response_schema=response_schema
        ),

        contents=content
    )


    return response.text


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(BaseModel):

    content: str

    question_type: str


class AnswerRequest(BaseModel):

    test_id: str

    question_number: int

    answer: str


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse("index.html")


# ============================================================
# GENERATE QUESTIONS
# ============================================================

@app.post("/generate")
def generate_questions(request: QuestionRequest):

    if not request.content.strip():

        return {
            "error": "Please enter your study content."
        }


    if request.question_type not in question_types:

        return {
            "error": "Invalid question type."
        }


    try:

        result = question_generator(
            request.content,
            request.question_type
        )


        print("\n================ AI RESPONSE ================\n")
        print(result)
        print("\n==============================================\n")


        questions_data = json.loads(result)


        # ====================================================
        # VALIDATE QUESTIONS
        # ====================================================

        if "questions" not in questions_data:

            return {
                "error": "Invalid question format returned by AI."
            }


        if len(questions_data["questions"]) != 5:

            return {
                "error": "AI did not generate exactly 5 questions."
            }


        # ====================================================
        # CREATE TEST ID
        # ====================================================

        test_id = str(uuid.uuid4())


        # Store complete questions internally
        tests[test_id] = questions_data


        # ====================================================
        # PREPARE PUBLIC QUESTIONS
        # ====================================================

        public_questions = []


        for question in questions_data["questions"]:

            # ----------------------------------------------
            # Normalize MCQ options
            # ----------------------------------------------

            normalized_options = []


            for index, option in enumerate(
                question.get("options", [])
            ):

                # Expected Gemini format
                if isinstance(option, dict):

                    letter = option.get(
                        "letter",
                        chr(65 + index)
                    )

                    text = option.get(
                        "text",
                        ""
                    )

                # Fallback if Gemini returns plain strings
                elif isinstance(option, str):

                    letter = chr(65 + index)

                    text = option

                else:

                    letter = chr(65 + index)

                    text = str(option)


                normalized_options.append({

                    "letter": letter,

                    "text": text
                })


            public_question = {

                "number":
                    question["number"],

                "question":
                    question["question"],

                "options":
                    normalized_options
            }


            public_questions.append(
                public_question
            )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "test_id":
                test_id,

            "question_type":
                request.question_type,

            "questions":
                public_questions
        }


    except json.JSONDecodeError as e:

        print("JSON ERROR:", e)

        return {

            "error":
            "The AI returned an invalid response. Please try again."
        }


    except Exception as e:

        print("ERROR:", e)

        return {

            "error":
            "Gemini is temporarily unavailable. Please try again."
        }


# ============================================================
# CHECK MCQ ANSWER
# ============================================================

@app.post("/check-answer")
def check_answer(request: AnswerRequest):

    test = tests.get(request.test_id)


    if not test:

        return {
            "error": "Test not found or expired."
        }


    question = None


    for item in test["questions"]:

        if item["number"] == request.question_number:

            question = item

            break


    if not question:

        return {
            "error": "Question not found."
        }


    user_answer = request.answer.upper().strip()


    correct_answer = (
        question["correct_answer"]
        .upper()
        .strip()
    )


    is_correct = (
        user_answer == correct_answer
    )


    correct_option_text = ""


    for option in question.get(
        "options",
        []
    ):

        if option.get("letter") == correct_answer:

            correct_option_text = option.get(
                "text",
                ""
            )

            break


    return {

        "correct":
            is_correct,

        "selected_answer":
            user_answer,

        "correct_answer":
            correct_answer,

        "correct_option_text":
            correct_option_text,

        "explanation":
            question.get(
                "explanation",
                ""
            )
    }


# ============================================================
# SUBMIT SHORT ANSWER / INTERVIEW ANSWER
# ============================================================

@app.post("/submit-answer")
def submit_answer(request: AnswerRequest):

    test = tests.get(request.test_id)


    if not test:

        return {
            "error": "Test not found or expired."
        }


    question = None


    for item in test["questions"]:

        if item["number"] == request.question_number:

            question = item

            break


    if not question:

        return {
            "error": "Question not found."
        }


    return {

        "user_answer":
            request.answer.strip(),

        "model_answer":
            question.get(
                "model_answer",
                ""
            ),

        "explanation":
            question.get(
                "explanation",
                ""
            ),

        "evaluation_points":
            question.get(
                "evaluation_points",
                []
            )
    }