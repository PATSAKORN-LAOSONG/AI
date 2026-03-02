import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Adaptive Learning System", layout="centered")
st.title("🤖 AI Adaptive Learning System (3 Subjects)")

# ======================================
# SESSION STATE
# ======================================
if "subject" not in st.session_state:
    st.session_state.subject = None

if "level" not in st.session_state:
    st.session_state.level = "medium"

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "score" not in st.session_state:
    st.session_state.score = 0

if "correct_streak" not in st.session_state:
    st.session_state.correct_streak = 0

if "wrong_streak" not in st.session_state:
    st.session_state.wrong_streak = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "finished" not in st.session_state:
    st.session_state.finished = False

if "chat" not in st.session_state:
    st.session_state.chat = []

levels = ["easy", "medium", "hard"]

# ======================================
# QUESTION BANK 3 SUBJECTS
# ======================================
question_bank = {

    "คณิตศาสตร์": {
        "easy": [
            {"q": "2 + 3 =", "choices": ["4","5","6","7"], "answer": "5"},
            {"q": "4 + 4 =", "choices": ["6","7","8","9"], "answer": "8"},
        ],
        "medium": [
            {"q": "อนุพันธ์ของ x^2 คืออะไร?", "choices": ["x","2x","x^2","2"], "answer": "2x"},
            {"q": "5^2 =", "choices": ["10","20","25","15"], "answer": "25"},
        ],
        "hard": [
            {"q": "อนุพันธ์ของ x^3 =", "choices": ["2x","3x^2","x^2","3x"], "answer": "3x^2"},
            {"q": "อินทิกรัลของ x^2 =", "choices": ["x^3/3","2x","x^2","3x"], "answer": "x^3/3"},
        ]
    },

    "ภาษาไทย": {
        "easy": [
            {"q": "คำใดเป็นคำนาม?", "choices": ["กิน","วิ่ง","โต๊ะ","เร็ว"], "answer": "โต๊ะ"},
            {"q": "คำว่า 'ดี' เป็นคำประเภทใด?", "choices": ["กริยา","วิเศษณ์","นาม","สรรพนาม"], "answer": "วิเศษณ์"},
        ],
        "medium": [
            {"q": "ข้อใดเป็นประโยคสมบูรณ์?", 
             "choices": ["แมวสีดำ","ฉันกินข้าวแล้ว","เร็วมาก","เพราะฝนตก"], 
             "answer": "ฉันกินข้าวแล้ว"},
            {"q": "คำว่า 'ประเทศไทย' เป็นคำประเภทใด?", 
             "choices": ["นามเฉพาะ","นามทั่วไป","กริยา","วิเศษณ์"], 
             "answer": "นามเฉพาะ"},
        ],
        "hard": [
            {"q": "ข้อใดเป็นประโยครวม?", 
             "choices": ["ฉันกินข้าว","ฉันไปโรงเรียนและอ่านหนังสือ","แมววิ่ง","ฝนตก"], 
             "answer": "ฉันไปโรงเรียนและอ่านหนังสือ"},
            {"q": "คำใดเป็นคำซ้อน?", 
             "choices": ["ดีใจ","บ้านเรือน","สวย","กิน"], 
             "answer": "บ้านเรือน"},
        ]
    },

    "ภาษาอังกฤษ": {
        "easy": [
            {"q": "Dog แปลว่าอะไร?", "choices": ["แมว","สุนัข","ปลา","นก"], "answer": "สุนัข"},
            {"q": "I ___ a student.", "choices": ["am","is","are","be"], "answer": "am"},
        ],
        "medium": [
            {"q": "She ___ to school every day.", 
             "choices": ["go","goes","going","gone"], 
             "answer": "goes"},
            {"q": "Past tense ของ 'eat' คืออะไร?", 
             "choices": ["eated","ate","eats","eating"], 
             "answer": "ate"},
        ],
        "hard": [
            {"q": "If I ___ rich, I would travel the world.", 
             "choices": ["am","was","were","be"], 
             "answer": "were"},
            {"q": "Choose the correct passive form: 'The book ___ by him.'", 
             "choices": ["writes","was written","is writing","written"], 
             "answer": "was written"},
        ]
    }
}

# ======================================
# ML MODEL
# ======================================
def train_model():
    data = {
        "score": [0,1,2,3,4,5],
        "level": ["easy","easy","medium","medium","hard","hard"]
    }
    df = pd.DataFrame(data)
    X = df[["score"]]
    y = df["level"]
    model = RandomForestClassifier()
    model.fit(X,y)
    return model

model = train_model()

# ======================================
# เลือกวิชา
# ======================================
if st.session_state.subject is None:
    subject = st.selectbox("เลือกวิชา", ["คณิตศาสตร์","ภาษาไทย","ภาษาอังกฤษ"])
    if st.button("เริ่มทำแบบทดสอบ"):
        st.session_state.subject = subject
        st.session_state.current_question = None
        st.rerun()

# ======================================
# QUIZ SYSTEM
# ======================================
if st.session_state.subject and not st.session_state.finished:

    subject = st.session_state.subject

    def get_question():
        return random.choice(
            question_bank[subject][st.session_state.level]
        )

    if st.session_state.current_question is None:
        st.session_state.current_question = get_question()

    q = st.session_state.current_question

    st.write(f"### วิชา: {subject}")
    st.write(f"ข้อที่ {st.session_state.question_count+1} (ระดับ {st.session_state.level.upper()})")
    st.write(q["q"])

    selected = st.radio("เลือกคำตอบ:", q["choices"], key="answer")

    if st.button("ส่งคำตอบ"):

        if selected == q["answer"]:
            st.session_state.score += 1
            st.session_state.correct_streak += 1
            st.session_state.wrong_streak = 0
            st.session_state.chat.append("✅ ตอบถูก")
        else:
            st.session_state.wrong_streak += 1
            st.session_state.correct_streak = 0
            st.session_state.chat.append("❌ ตอบผิด")

        # Adaptive
        if st.session_state.correct_streak == 2:
            if st.session_state.level != "hard":
                idx = levels.index(st.session_state.level)
                st.session_state.level = levels[idx+1]
            st.session_state.correct_streak = 0

        if st.session_state.wrong_streak == 2:
            if st.session_state.level != "easy":
                idx = levels.index(st.session_state.level)
                st.session_state.level = levels[idx-1]
            st.session_state.wrong_streak = 0

        st.session_state.question_count += 1

        if st.session_state.question_count >= 5:
            st.session_state.finished = True
        else:
            st.session_state.current_question = get_question()

        st.rerun()

# ======================================
# RESULT
# ======================================
if st.session_state.finished:

    st.success(f"คะแนนของคุณ: {st.session_state.score}/5")

    predicted = model.predict([[st.session_state.score]])[0]
    st.info(f"🤖 AI ประเมินระดับคุณเป็น: {predicted.upper()}")

    st.write("💬 สรุปผล:")
    for msg in st.session_state.chat:
        st.write(msg)

    if st.button("เริ่มใหม่"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
