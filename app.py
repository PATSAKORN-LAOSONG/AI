# app.py
import streamlit as st
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Math Skill AI", layout="centered")
st.title("🧠 ระบบวิเคราะห์จุดอ่อนคณิตศาสตร์ด้วย AI")

# -----------------------
# ช่วยผู้ใช้: อัพโหลด dataset หรือใช้ไฟล์ในเครื่อง
# -----------------------
st.sidebar.header("ข้อมูล (Dataset)")
uploaded_file = st.sidebar.file_uploader(
    "อัปโหลดไฟล์ CSV (คอลัมน์: addition, subtraction, multiplication, division, label)",
    type=["csv"],
)

use_class_weight = st.sidebar.checkbox("ฝึกโมเดลด้วย class_weight='balanced' (แก้ imbalance)", value=True)
retrain_button = st.sidebar.button("🔁 Retrain model (ใช้ dataset ปัจจุบัน)")

# -----------------------
# ฟังก์ชันช่วยโหลด dataset (จาก upload หรือไฟล์ในเครื่องหรือสร้างตัวอย่าง)
# -----------------------
def get_dataset(uploaded):
    # ถ้ามีอัปโหลด ใช้อันนั้น
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("อ่านไฟล์ที่อัปโหลดสำเร็จ")
            return df, "uploaded"
        except Exception as e:
            st.sidebar.error(f"อ่านไฟล์ที่อัปโหลดไม่สำเร็จ: {e}")
            # ตกไปสร้างตัวอย่าง
    # ถ้าไม่มีไฟล์อัปโหลด ลองอ่านจากไดเรกทอรีงาน
    local_path = "math_skill_dataset_200.csv"
    try:
        df = pd.read_csv(local_path)
        st.sidebar.success(f"อ่านไฟล์จากเครื่อง: {local_path}")
        return df, "local"
    except Exception:
        # สร้าง dataset ตัวอย่างแบบ synthetic (balanced)
        st.sidebar.info("ไม่มีไฟล์ dataset พบ — จะใช้ตัวอย่าง synthetic แทน (เพื่อทดสอบ)")
        rng = np.random.RandomState(42)
        N = 200
        # สุ่มคะแนนเป็น 0-100
        addition = rng.randint(0, 101, size=N)
        subtraction = rng.randint(0, 101, size=N)
        multiplication = rng.randint(0, 101, size=N)
        division = rng.randint(0, 101, size=N)
        labels = []
        for a, s, m, d in zip(addition, subtraction, multiplication, division):
            scores = {"add": a, "sub": s, "mul": m, "div": d}
            min_key = min(scores, key=scores.get)
            # ถ้าทุกคะแนนดีมาก ให้ strong_all
            if min(scores.values()) >= 75:
                lbl = "strong_all"
            else:
                lbl = f"weak_{min_key}"
            labels.append(lbl)
        df = pd.DataFrame({
            "addition": addition,
            "subtraction": subtraction,
            "multiplication": multiplication,
            "division": division,
            "label": labels
        })
        return df, "synthetic"

# โหลด dataset (และเก็บไว้ใน session_state เพื่อเรียกใช้ต่อ)
if "df_source" not in st.session_state or retrain_button:
    df, src = get_dataset(uploaded_file)
    st.session_state.df = df
    st.session_state.df_source = src
else:
    df = st.session_state.df
    src = st.session_state.df_source

# -----------------------
# ฟังก์ชันฝึกโมเดล
# -----------------------
@st.cache_resource
def train_model_cached(df_serialized, class_weight_flag):
    # ฟังก์ชันนี้รับ df_serialized เป็นค่า hashable (string) เพื่อให้ st.cache_resource ทำงานได้
    # การแปลงกลับเป็น DataFrame
    df_local = pd.read_json(df_serialized, orient="split")
    X = df_local[["addition", "subtraction", "multiplication", "division"]]
    y = df_local["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if class_weight_flag:
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# เตรียม df_serialized สำหรับ caching
try:
    df_serialized = df.to_json(orient="split")
except Exception:
    # กรณี dataframe ใหญ่มากและไม่สามารถ serialize (แทบจะไม่เกิด) -> fallback ใช้ CSV string
    df_serialized = df.to_csv(index=False)

# ฝึก/โหลดโมเดล
with st.spinner("กำลังฝึกรุ่น (model training)..."):
    model = train_model_cached(df_serialized, use_class_weight)

# -----------------------
# ฟังก์ชันสร้างโจทย์
# -----------------------
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

    choices = set()
    choices.add(correct)
    while len(choices) < 4:
        delta = random.choice([1,2,3,4,5,6,7,8,9])
        sign = random.choice([1, -1])
        cand = correct + sign * delta
        if cand >= 0:
            choices.add(cand)
    choices = list(choices)
    random.shuffle(choices)
    return question, correct, choices

# -----------------------
# เตรียมข้อสอบ 12 ข้อ (session state)
# -----------------------
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

# -----------------------
# แสดงข้อสอบ
# -----------------------
st.subheader("📘 ทำแบบทดสอบ 12 ข้อ")
st.caption("โปรดตอบทุกข้อ แล้วกดปุ่ม 'ส่งคำตอบ'")

PLACEHOLDER = "-- เลือกคำตอบ --"
if "user_answers" not in st.session_state:
    st.session_state.user_answers = [PLACEHOLDER] * len(st.session_state.questions)

for i, q in enumerate(st.session_state.questions):
    choices_with_placeholder = [PLACEHOLDER] + [str(c) for c in q["choices"]]
    selected = st.radio(f"ข้อ {i+1}: {q['question']}", choices_with_placeholder, key=f"q{i}")
    st.session_state.user_answers[i] = selected

# -----------------------
# ตรวจคำตอบ
# -----------------------
if st.button("ส่งคำตอบ"):
    if any(ans == PLACEHOLDER for ans in st.session_state.user_answers):
        st.warning("กรุณาตอบให้ครบทุกข้อก่อนส่ง (ยังมีข้อที่ไม่ได้เลือกคำตอบ).")
    else:
        # นับคะแนน per operation
        scores = {"add":0, "sub":0, "mul":0, "div":0}
        for i, q in enumerate(st.session_state.questions):
            user_val = st.session_state.user_answers[i]
            try:
                user_val_num = int(user_val)
            except:
                user_val_num = None
            if user_val_num == q["answer"]:
                scores[q["operation"]] += 1

        # Normalize 0–100
        add_score = round((scores["add"]/3)*100,2)
        sub_score = round((scores["sub"]/3)*100,2)
        mul_score = round((scores["mul"]/3)*100,2)
        div_score = round((scores["div"]/3)*100,2)

        # เก็บลง session เพื่อให้ Debug อ่านค่าได้
        st.session_state.last_scores = {
            "add": add_score,
            "sub": sub_score,
            "mul": mul_score,
            "div": div_score
        }

        st.subheader("📊 ผลคะแนน")
        st.write(f"➕ การบวก: {add_score}")
        st.write(f"➖ การลบ: {sub_score}")
        st.write(f"✖ การคูณ: {mul_score}")
        st.write(f"➗ การหาร: {div_score}")

        # ถ้าได้เต็มทุกหมวด
        if add_score == 100 and sub_score == 100 and mul_score == 100 and div_score == 100:
            st.success("🎉 คุณพร้อมเรียนบทต่อไปแล้ว!")
        else:
            # ใช้โมเดลทำนาย
            X_in = [[add_score, sub_score, mul_score, div_score]]
            try:
                prediction = model.predict(X_in)
                result = prediction[0]
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดขณะทำนายด้วยโมเดล: {e}")
                result = "model_error"

            st.subheader("🤖 ผลการวิเคราะห์จาก AI (model prediction)")
            st.info(f"โมเดลทำนายจุดที่ควรพัฒนา: {result}")

            # --- วิเคราะห์จากคะแนนจริง (tie-aware) ---
            skill_scores = {
                "add": add_score,
                "sub": sub_score,
                "mul": mul_score,
                "div": div_score
            }
            min_score = min(skill_scores.values())
            weakest = [k for k, v in skill_scores.items() if v == min_score]
            friendly = {
                "add": ("การบวก", "https://www.youtube.com/watch?v=c5eS7nRsE_Q"),
                "sub": ("การลบ", "https://www.youtube.com/watch?v=vT_VBLlvdn8"),
                "mul": ("การคูณ", "https://www.youtube.com/watch?v=73obrcsERe8"),
                "div": ("การหาร", "https://www.youtube.com/watch?v=9D1JW8rYqeA")
            }

            st.subheader("🔎 วิเคราะห์จากคะแนนจริง (tie-aware)")
            st.write(f"คะแนนต่ำสุดคือ {min_score} — หัวข้อที่คะแนนต่ำสุด (จุดอ่อน):")
            for w in weakest:
                name, vid = friendly[w]
                st.write(f"- {name} (คะแนน {skill_scores[w]} / 100)")
                st.write(f"  → คำแนะนำ: ฝึก {name} เพิ่มเติม")
                st.video(vid)

            # แสดงหมายเหตุเมื่อโมเดลชี้อีกหัวข้อ
            if result not in [f"weak_{w}" for w in weakest] and result != "strong_all" and result != "model_error":
                st.write("")  # spacer
                st.write("หมายเหตุ: โมเดลยังชี้ไปที่:", result)

            # แสดง predict_proba (ถ้ามี) แบบสั้น ๆ
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X_in)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({"label": classes, "probability": probs})
                    prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
                    st.subheader("📈 ความน่าจะเป็นของแต่ละคลาส (model confidence)")
                    st.write(prob_df)
                    top_label = prob_df.iloc[0]["label"]
                    top_prob = prob_df.iloc[0]["probability"]
                    if top_prob < 0.5:
                        st.warning("ความมั่นใจของโมเดลไม่สูง — ให้ยึดการวิเคราะห์จากคะแนนจริง (tie-aware) เป็นหลัก")
                except Exception as e:
                    st.write("ไม่สามารถคำนวณ predict_proba ได้:", e)

# -----------------------
# Debug / Model insights expander
# -----------------------
with st.expander("🔧 Debug / Model insights (สำหรับพัฒนา)", expanded=False):
    st.write(f"Dataset source: **{src}** (ขนาด: {df.shape})")
    st.subheader("1) การแจกแจงของ label (value_counts)")
    try:
        vc = df["label"].value_counts()
        st.write(vc)
    except Exception as e:
        st.write("ไม่สามารถอ่านคอลัมน์ label ได้:", e)

    st.markdown("---")
    st.subheader("2) Feature importances (ถ้ามี)")
    feat_names = ["addition", "subtraction", "multiplication", "division"]
    if hasattr(model, "feature_importances_"):
        try:
            fi = model.feature_importances_
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
            st.write(fi_df)
            st.bar_chart(fi_df.set_index("feature")["importance"])
        except Exception as e:
            st.write("เกิดปัญหาในการดึง feature_importances_:", e)
    else:
        st.info("โมเดลไม่มี attribute 'feature_importances_'")

    st.markdown("---")
    st.subheader("3) ความน่าจะเป็นของการทำนาย (predict_proba) สำหรับเคสล่าสุด")
    last = st.session_state.get("last_scores", None)
    if last is None:
        st.info("ยังไม่มีผลคะแนนที่คำนวณ — ส่งคำตอบก่อนจึงจะเห็น predict_proba")
    else:
        X_last = [[last["add"], last["sub"], last["mul"], last["div"]]]
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_last)[0]
                classes = model.classes_
                prob_df = pd.DataFrame({"label": classes, "probability": probs})
                prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
                st.write(prob_df)
                st.write(f"Top prediction: **{prob_df.iloc[0]['label']}** (confidence {prob_df.iloc[0]['probability']:.2f})")
            except Exception as e:
                st.write("เกิดข้อผิดพลาดขณะคำนวณ predict_proba():", e)
        else:
            st.info("โมเดลนี้ไม่มีเมธอด predict_proba()")

# -----------------------
# ปุ่มเริ่มใหม่
# -----------------------
st.markdown("---")
if st.button("🔄 เริ่มใหม่ (reset quiz)"):
    keys_to_clear = ["questions", "user_answers", "last_scores"]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # รีโหลดหน้า
    try:
        st.experimental_rerun()
    except Exception:
        st.stop()

# -----------------------
# ข้อความช่วยเหลือท้ายหน้า
# -----------------------
st.write("---")
st.caption("หมายเหตุ: หากคุณเห็น FileNotFoundError เมื่อเรียกใช้งาน ให้ลองอัปโหลดไฟล์ CSV ในแถบด้านข้าง หรือวางไฟล์ math_skill_dataset_200.csv ไว้ในโฟลเดอร์เดียวกับไฟล์นี้")
