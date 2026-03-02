import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Learning Chatbot", layout="centered")

st.title("🤖 AI ระบบแนะนำบทเรียน")

# ==============================
# SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "score" not in st.session_state:
    st.session_state.score = 0

if "question_index" not in st.session_state:
    st.session_state.question_index = 0

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "finished" not in st.session_state:
    st.session_state.finished = False


# ==============================
# QUESTION BANK
# ==============================
question_bank = {
    "easy": [
        {"q": "2 + 3 =", "choices": ["4", "5", "6", "7"], "answer": "5"},
        {"q": "5 - 2 =", "choices": ["1", "2", "3", "4"], "answer": "3"},
        {"q": "3 x 2 =", "choices": ["5", "6", "7", "8"], "answer": "6"},
        {"q": "10 ÷ 2 =", "choices": ["3", "4", "5", "6"], "answer": "5"},
        {"q": "4 + 4 =", "choices": ["6", "7", "8", "9"], "answer": "8"},
        {"q": "6 - 1 =", "choices": ["4", "5", "6", "7"], "answer": "5"},
        {"q": "1 + 9 =", "choices": ["9", "10", "11", "12"], "answer": "10"}
    ],
    "medium": [
        {"q": "อนุพันธ์ของ x^2 คืออะไร?", "choices": ["x", "2x", "x^2", "2"], "answer": "2x"},
        {"q": "อนุพันธ์ของ 3x คืออะไร?", "choices": ["3", "x", "6x", "1"], "answer": "3"},
        {"q": "lim x→0 ของ x^2 =", "choices": ["0", "1", "2", "ไม่มีค่า"], "answer": "0"},
        {"q": "อินทิกรัลของ 2x =", "choices": ["x^2", "2", "x", "x^2+1"], "answer": "x^2"},
        {"q": "√16 =", "choices": ["2", "3", "4", "5"], "answer": "4"},
        {"q": "5^2 =", "choices": ["10", "20", "25", "15"], "answer": "25"}
    ],
    "hard": [
        {"q": "อนุพันธ์ของ x^3 =", "choices": ["2x", "3x^2", "x^2", "3x"], "answer": "3x^2"},
        {"q": "lim x→∞ ของ 1/x =", "choices": ["0", "1", "∞", "-∞"], "answer": "0"},
        {"q": "อินทิกรัลของ x^2 =", "choices": ["x^3/3", "2x", "x^2", "3x"], "answer": "x^3/3"},
        {"q": "sin(90°) =", "choices": ["0", "1", "-1", "0.5"], "answer": "1"},
        {"q": "cos(0°) =", "choices": ["0", "1", "-1", "0.5"], "answer": "1"},
        {"q": "log10(100) =", "choices": ["1", "2", "10", "100"], "answer": "2"}
    ]
}


# ==============================
# RANDOM 5 QUESTIONS EACH LEVEL
# ==============================
def generate_questions():
    questions = []
    for level in question_bank:
        questions.extend(random.sample(question_bank[level], 5))
    random.shuffle(questions)
    return questions


# ==============================
# ML MODEL (Classification)
# ==============================
def train_model():
    data = {
        "score": [0, 2, 4, 6, 8, 10, 12, 14, 15],
        "level": ["easy", "easy", "easy",
                  "medium", "medium", "medium",
                  "hard", "hard", "hard"]
    }
    df = pd.DataFrame(data)

    X = df[["score"]]
    y = df["level"]

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()


# ==============================
# START BUTTON
# ==============================
if st.button("🚀 เริ่มทำแบบทดสอบใหม่"):
    st.session_state.questions = generate_questions()
    st.session_state.score = 0
    st.session_state.question_index = 0
    st.session_state.answers = {}
    st.session_state.finished = False
    st.rerun()


# ==============================
# SHOW QUESTIONS
# ==============================
if st.session_state.questions and not st.session_state.finished:

    q_index = st.session_state.question_index
    question = st.session_state.questions[q_index]

    st.subheader(f"ข้อที่ {q_index + 1}")
    st.write(question["q"])

    selected = st.radio(
        "เลือกคำตอบ:",
        question["choices"],
        key=f"q_{q_index}"
    )

    if st.button("ส่งคำตอบ"):

        if selected == question["answer"]:
            st.session_state.score += 1

        st.session_state.question_index += 1

        if st.session_state.question_index >= len(st.session_state.questions):
            st.session_state.finished = True

        st.rerun()


# ==============================
# RESULT + AI CLASSIFICATION
# ==============================
if st.session_state.finished:

    st.success(f"คุณได้คะแนนทั้งหมด {st.session_state.score} / 15")

    predicted_level = model.predict([[st.session_state.score]])[0]

    st.info(f"🤖 AI ประเมินระดับคุณเป็น: {predicted_level.upper()}")

    if predicted_level == "easy":
        st.write("แนะนำให้ทบทวนพื้นฐานเพิ่มเติม")
    elif predicted_level == "medium":
        st.write("คุณอยู่ระดับกลาง ฝึกต่ออีกนิดจะเก่งมาก!")
    else:
        st.write("ยอดเยี่ยม! คุณอยู่ระดับสูง 🎯")
