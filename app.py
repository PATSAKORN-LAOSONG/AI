import streamlit as st
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Math Skill AI", layout="centered")
st.title("🧠 ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์ด้วย AI")

# =========================
# โหลด Dataset และ Train ML
# =========================
@st.cache_resource
def train_model():
    df = pd.read_csv("math_skill_dataset_200.csv")

    X = df[["addition", "subtraction", "multiplication", "division"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, df

model, df = train_model()

# =========================
# ฟังก์ชันสร้างโจทย์
# =========================
def generate_question(operation):
    if operation == "add":
        a, b = random.randint(1, 50), random.randint(1, 50)
        correct = a + b
        question = f"{a} + {b} = ?"

    elif operation == "sub":
        a, b = random.randint(1, 50), random.randint(1, 50)
        if a < b:
            a, b = b, a
        correct = a - b
        question = f"{a} - {b} = ?"

    elif operation == "mul":
        a, b = random.randint(1, 12), random.randint(1, 12)
        correct = a * b
        question = f"{a} × {b} = ?"

    elif operation == "div":
        b = random.randint(1, 12)
        correct = random.randint(1, 12)
        a = b * correct
        question = f"{a} ÷ {b} = ?"

    choices = set([correct])
    while len(choices) < 4:
        delta = random.choice([1,2,3,4,5,6,7,8,9])
        sign = random.choice([1, -1])
        cand = correct + sign * delta
        if cand >= 0:
            choices.add(cand)

    choices = list(choices)
    random.shuffle(choices)
    return question, correct, choices

# =========================
# เตรียมข้อสอบ 12 ข้อ
# =========================
if "questions" not in st.session_state:
    st.session_state.questions = []
    operations = ["add"]*3 + ["sub"]*3 + ["mul"]*3 + ["div"]*3
    random.shuffle(operations)

    for op in operations:
        q, ans, choices = generate_question(op)
        st.session_state.questions.append({
            "operation": op,
            "question": q,
            "answer": ans,
            "choices": choices
        })

# =========================
# แสดงข้อสอบ
# =========================
scores = {"add":0, "sub":0, "mul":0, "div":0}

st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
st.caption("โปรดตอบทุกข้อ แล้วกดปุ่ม 'ส่งคำตอบ'")

PLACEHOLDER = "-- เลือกคำตอบ --"

if "user_answers" not in st.session_state:
    st.session_state.user_answers = [PLACEHOLDER] * len(st.session_state.questions)

for i, q in enumerate(st.session_state.questions):
    choices_with_placeholder = [PLACEHOLDER] + [str(c) for c in q["choices"]]
    selected = st.radio(
        f"ข้อ {i+1}: {q['question']}",
        choices_with_placeholder,
        key=f"q{i}"
    )
    st.session_state.user_answers[i] = selected

# =========================
# ตรวจคำตอบ
# =========================
if st.button("ส่งคำตอบ"):
    if any(ans == PLACEHOLDER for ans in st.session_state.user_answers):
        st.warning("กรุณาตอบให้ครบทุกข้อก่อนส่ง")
    else:
        # นับคะแนน
        for i, q in enumerate(st.session_state.questions):
            user_val = st.session_state.user_answers[i]
            try:
                user_val_num = int(user_val)
            except:
                user_val_num = None

            if user_val_num == q["answer"]:
                scores[q["operation"]] += 1

        # แปลงเป็น 0–100 (ต่อหมวดมี 3 ข้อ)
        add_score = round((scores["add"]/3)*100, 2)
        sub_score = round((scores["sub"]/3)*100, 2)
        mul_score = round((scores["mul"]/3)*100, 2)
        div_score = round((scores["div"]/3)*100, 2)

        st.subheader("📊 ผลคะแนน")
        st.write(f"➕ การบวก: {add_score}")
        st.write(f"➖ การลบ: {sub_score}")
        st.write(f"✖ การคูณ: {mul_score}")
        st.write(f"➗ การหาร: {div_score}")

        # -------------------------
        # วิเคราะห์จากคะแนนจริง (หลัก)
        # -------------------------
        skill_scores = {
            "add": add_score,
            "sub": sub_score,
            "mul": mul_score,
            "div": div_score
        }

        friendly = {
            "add": ("การบวก", "https://www.youtube.com/watch?v=c5eS7nRsE_Q"),
            "sub": ("การลบ", "https://www.youtube.com/watch?v=vT_VBLlvdn8"),
            "mul": ("การคูณ", "https://www.youtube.com/watch?v=73obrcsERe8"),
            "div": ("การหาร", "https://www.youtube.com/watch?v=9D1JW8rYqeA")
        }

        # เกณฑ์ผ่าน: ถ้าทุกทักษะ >= 60 ถือว่าพื้นฐานดี
        pass_threshold = 60

        # หา min + tie-aware
        min_score = min(skill_scores.values())
        weakest = [k for k, v in skill_scores.items() if v == min_score]

        st.subheader("🔎 วิเคราะห์จากคะแนนจริง (แนะนำหลัก)")

        if all(score >= pass_threshold for score in skill_scores.values()):
            st.success("พื้นฐานคณิตศาสตร์อยู่ในระดับดี ไม่มีจุดอ่อนที่ชัดเจน 🎉")
        else:
            st.write(f"คะแนนต่ำสุดคือ {min_score} — หัวข้อที่ควรพัฒนา:")
            for w in weakest:
                name, vid = friendly[w]
                st.write(f"- {name} (คะแนน {skill_scores[w]} / 100)")
                st.write(f"→ คำแนะนำ: ฝึก{name}เพิ่มเติม")
                st.video(vid)

        # -------------------------
        # AI Prediction (เสริม)
        # -------------------------
        st.subheader("🤖 ผลการวิเคราะห์จาก AI (เสริม)")

        # ทำนาย + ความน่าจะเป็น
        X_input = [[add_score, sub_score, mul_score, div_score]]
        result = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        classes = list(model.classes_)
        proba_map = {c: float(p) for c, p in zip(classes, proba)}

        st.info(f"โมเดลทำนายจุดที่ควรพัฒนา: {result}")

        # map label -> skill key
        model_map = {
            "weak_add": "add",
            "weak_sub": "sub",
            "weak_mul": "mul",
            "weak_div": "div",
        }
        predicted_skill = model_map.get(result)

        # กฎใหม่: ถ้าโมเดลทายไม่ตรงกับ “จุดอ่อนจากคะแนนจริง”
        # ให้แจ้งว่าใช้คะแนนจริงเป็นหลัก และโมเดลเป็นเสริม
        if predicted_skill and predicted_skill not in weakest and not all(score >= pass_threshold for score in skill_scores.values()):
            st.warning(
                "หมายเหตุ: กรณีนี้คะแนนจริงชี้ชัดว่าควรพัฒนาหัวข้อที่คะแนนต่ำสุดก่อน "
                "ส่วนผลจากโมเดลเป็นการทำนายจากรูปแบบในชุดข้อมูล จึงอาจไม่ตรงกับคะแนนจริงได้"
            )

        # แสดงความน่าจะเป็น (ช่วยอธิบายว่าทำไมโมเดลเลือกอันนั้น)
        with st.expander("ดูความน่าจะเป็นของโมเดล (predict_proba)"):
            # เรียงจากมากไปน้อย
            sorted_items = sorted(proba_map.items(), key=lambda x: x[1], reverse=True)
            st.write("ความน่าจะเป็นที่โมเดลให้แต่ละคลาส:")
            for c, p in sorted_items:
                st.write(f"- {c}: {round(p*100, 2)}%")

# =========================
# ปุ่มเริ่มใหม่
# =========================
if st.button("🔄 เริ่มใหม่"):
    keys_to_clear = ["questions", "user_answers"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

    try:
        st.experimental_rerun()
    except Exception:
        st.stop()
