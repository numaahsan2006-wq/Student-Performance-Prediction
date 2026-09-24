
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="EduPredict | Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "StudentPerformanceFactors.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_PATH = ROOT / "models" / "model_results.csv"

FEATURES = [
    "Hours_Studied","Attendance","Parental_Involvement",
    "Access_to_Resources","Extracurricular_Activities","Sleep_Hours",
    "Previous_Scores","Motivation_Level","Internet_Access",
    "Tutoring_Sessions","Family_Income","Teacher_Quality",
    "School_Type","Peer_Influence","Physical_Activity",
    "Learning_Disabilities","Parental_Education_Level",
    "Distance_from_Home","Gender","Mathematics_Score",
    "Science_Score","English_Score","Internal_Marks","Participation",
]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip().str.replace(" ","_",regex=False)
    for c in list(df.columns):
        if c.lower().replace("_","") == "examscore":
            df = df.rename(columns={c:"Exam_Score"})
    df["Exam_Score"] = pd.to_numeric(df["Exam_Score"], errors="coerce")
    return df.dropna(subset=["Exam_Score"]).copy()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

def level(score):
    if score >= 80: return "High", "🟢", "high"
    if score >= 60: return "Average", "🟡", "average"
    return "Low", "🔴", "low"

def recommendations(a):
    r=[]
    if a["Attendance"] < 75: r.append("Improve attendance and maintain a consistent learning routine.")
    elif a["Attendance"] < 85: r.append("Aim for 85%+ attendance to improve learning continuity.")
    if a["Hours_Studied"] < 15: r.append("Increase focused study time gradually.")
    if a["Mathematics_Score"] < 60: r.append("Focus on Mathematics through targeted practice.")
    if a["Science_Score"] < 60: r.append("Focus on Science through additional practice sessions.")
    if a["English_Score"] < 60: r.append("Focus on English through reading and written practice.")
    if a["Internal_Marks"] < 60: r.append("Improve internal assessment performance through regular assignments and revision.")
    if a["Participation"] < 60: r.append("Increase classroom participation and practice activities.")
    if a["Tutoring_Sessions"] == 0: r.append("Consider additional tutoring or guided practice for difficult topics.")
    if a["Sleep_Hours"] < 6: r.append("Improve sleep consistency to support concentration.")
    if not r: r.append("Maintain the current learning habits and continue regular practice.")
    return r[:6]

st.html("""
<style>
.stApp{background:#000;color:#f8fafc}
.block-container{max-width:1500px;padding-top:1.2rem}
.hero{padding:42px;border-radius:28px;background:radial-gradient(circle at 85% 20%,rgba(34,211,238,.13),transparent 25%),radial-gradient(circle at 70% 90%,rgba(217,70,239,.14),transparent 30%),linear-gradient(135deg,#05050d,#0b0820,#030307);border:1px solid rgba(139,92,246,.6);box-shadow:0 0 55px rgba(124,58,237,.12);animation:in .6s ease}
@keyframes in{from{opacity:0;transform:translateY(15px)}to{opacity:1;transform:translateY(0)}}
.hero h1{font-size:clamp(2.3rem,5vw,4rem);font-weight:900;line-height:1.05}.grad{background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6,#60a5fa);background-size:250%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:g 5s linear infinite}@keyframes g{to{background-position:250%}}
.badge{display:inline-block;padding:7px 13px;border-radius:99px;border:1px solid rgba(167,139,250,.35);color:#c4b5fd;background:rgba(139,92,246,.1);font-size:.7rem;font-weight:800;letter-spacing:.1em}
.hero p{color:#a1a1aa;max-width:760px;line-height:1.7}.hero-icon{position:absolute;right:8%;margin-top:-100px;font-size:7rem;filter:drop-shadow(0 0 18px rgba(139,92,246,.8));animation:f 4s ease-in-out infinite}@keyframes f{50%{transform:translateY(-12px) rotate(3deg)}}
.section{margin:28px 0 14px}.section-title{font-size:1.35rem;font-weight:800}.section-sub{color:#71717a;font-size:.82rem}
.card,.metric,.result{padding:22px;border-radius:20px;background:linear-gradient(145deg,rgba(17,17,27,.94),rgba(5,5,10,.96));border:1px solid rgba(255,255,255,.08)}
.metric{min-height:125px;transition:.25s}.metric:hover{transform:translateY(-5px);border-color:rgba(139,92,246,.5);box-shadow:0 0 30px rgba(124,58,237,.12)}
.metric-icon{font-size:1.4rem}.metric-label{color:#71717a;font-size:.76rem;margin-top:7px}.metric-value{font-size:1.9rem;font-weight:900}
.result{text-align:center;padding:35px;background:radial-gradient(circle,rgba(124,58,237,.16),transparent 65%),#09090f;border-color:rgba(139,92,246,.45)}
.score{font-size:5rem;font-weight:900;background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.high{color:#34d399}.average{color:#fbbf24}.low{color:#fb7185}
.rec{padding:14px 18px;margin:8px 0;border-radius:13px;background:rgba(124,58,237,.08);border-left:3px solid #8b5cf6;color:#c4b5fd}
.stButton>button{border-radius:13px!important;background:linear-gradient(90deg,#7c3aed,#2563eb)!important;color:#fff!important;font-weight:800!important;min-height:48px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#050509,#090812,#020204)!important;border-right:1px solid rgba(139,92,246,.25)}
</style>
""")

st.html('<div style="text-align:center;font-size:3rem">🎓</div><div style="text-align:center;font-size:1.7rem;font-weight:900;background:linear-gradient(90deg,#60a5fa,#a78bfa,#e879f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent">EduPredict</div><div style="text-align:center;color:#71717a;font-size:.72rem">Student Performance Intelligence</div>')

page=st.sidebar.radio("Navigation",["🔮 Predict Student","🏠 Dashboard","📊 Analytics","🤖 Model Performance"],index=0)

if page=="🔮 Predict Student":
    st.html('<div class="hero"><div class="badge">✦ INDIVIDUAL STUDENT PREDICTION</div><h1>Predict Student <span class="grad">Performance</span></h1><p>Enter academic, subject and participation data to predict the exam score, classify performance, compare the student with the dataset, identify weak subjects and generate recommendations.</p><div class="hero-icon">🔮</div></div>')

    st.html('<div class="section"><div class="section-title">📚 Academic Profile</div><div class="section-sub">Core academic indicators</div></div>')
    c1,c2,c3=st.columns(3)
    with c1: hours=st.slider("Hours Studied",1,50,20)
    with c2: attendance=st.slider("Attendance (%)",0,100,80)
    with c3: previous=st.slider("Previous Score",0,100,70)
    c1,c2,c3=st.columns(3)
    with c1: tutoring=st.slider("Tutoring Sessions",0,10,2)
    with c2: sleep=st.slider("Sleep Hours",1,12,7)
    with c3: physical=st.slider("Physical Activity",0,10,3)

    st.html('<div class="section"><div class="section-title">📖 Subject & Internal Performance</div><div class="section-sub">Subject-level indicators required for academic analysis</div></div>')
    c1,c2,c3=st.columns(3)
    with c1: math=st.slider("Mathematics Score",0,100,70)
    with c2: science=st.slider("Science Score",0,100,72)
    with c3: english=st.slider("English Score",0,100,68)
    c1,c2=st.columns(2)
    with c1: internal=st.slider("Internal Marks",0,100,70)
    with c2: participation=st.slider("Participation (%)",0,100,70)

    st.html('<div class="section"><div class="section-title">🧠 Learning Environment</div><div class="section-sub">Family, school and learning factors</div></div>')
    c1,c2,c3=st.columns(3)
    with c1: parental=st.selectbox("Parental Involvement",["Low","Medium","High"],1)
    with c2: resources=st.selectbox("Access to Resources",["Low","Medium","High"],1)
    with c3: motivation=st.selectbox("Motivation Level",["Low","Medium","High"],1)
    c1,c2,c3=st.columns(3)
    with c1: extra=st.selectbox("Extracurricular Activities",["Yes","No"])
    with c2: internet=st.selectbox("Internet Access",["Yes","No"])
    with c3: income=st.selectbox("Family Income",["Low","Medium","High"],1)

    c1,c2,c3=st.columns(3)
    with c1: teacher=st.selectbox("Teacher Quality",["Low","Medium","High"],1)
    with c2: school=st.selectbox("School Type",["Public","Private"])
    with c3: peer=st.selectbox("Peer Influence",["Negative","Neutral","Positive"],1)
    c1,c2,c3=st.columns(3)
    with c1: disability=st.selectbox("Learning Disability",["Yes","No"],1)
    with c2: pedu=st.selectbox("Parental Education",["High School","College","Postgraduate"],1)
    with c3: distance=st.selectbox("Distance From Home",["Near","Moderate","Far"],1)
    gender=st.selectbox("Gender",["Male","Female"])

    if st.button("✨  PREDICT STUDENT PERFORMANCE",use_container_width=True):
        student={
            "Hours_Studied":hours,"Attendance":attendance,"Parental_Involvement":parental,
            "Access_to_Resources":resources,"Extracurricular_Activities":extra,"Sleep_Hours":sleep,
            "Previous_Scores":previous,"Motivation_Level":motivation,"Internet_Access":internet,
            "Tutoring_Sessions":tutoring,"Family_Income":income,"Teacher_Quality":teacher,
            "School_Type":school,"Peer_Influence":peer,"Physical_Activity":physical,
            "Learning_Disabilities":disability,"Parental_Education_Level":pedu,
            "Distance_from_Home":distance,"Gender":gender,"Mathematics_Score":math,
            "Science_Score":science,"English_Score":english,"Internal_Marks":internal,
            "Participation":participation,
        }
        pred=float(np.clip(model.predict(pd.DataFrame([student],columns=FEATURES))[0],0,100))
        lev,emoji,cls=level(pred)
        pct=(df["Exam_Score"]<=pred).mean()*100
        st.markdown("")
        st.html(f'<div class="result"><div style="color:#71717a;font-size:.8rem;letter-spacing:.1em">PREDICTED EXAM SCORE</div><div class="score">{pred:.1f}</div><div style="color:#71717a;font-size:.8rem">OUT OF 100</div><div class="{cls}" style="margin-top:18px;font-size:1.45rem;font-weight:900">{emoji} {lev.upper()} PERFORMANCE</div></div>')

        avg=float(df["Exam_Score"].mean())
        similar=df[(df["Attendance"].between(max(0,attendance-5),min(100,attendance+5))) & (df["Hours_Studied"].between(max(0,hours-5),hours+5))]
        st.html('<div class="section"><div class="section-title">📊 Student Comparison</div><div class="section-sub">Compare this prediction with the available student population</div></div>')
        cols=st.columns(4)
        vals=[("🎯","Predicted Score",f"{pred:.1f}"),("📚","Dataset Average",f"{avg:.1f}"),("👥","Similar Students",f"{len(similar):,}"),("📈","Dataset Percentile",f"{pct:.0f}%")]
        for col,(ico,lab,val) in zip(cols,vals):
            with col: st.html(f'<div class="metric"><div class="metric-icon">{ico}</div><div class="metric-label">{lab}</div><div class="metric-value">{val}</div></div>')

        st.html('<div class="section"><div class="section-title">📖 Subject-wise Analysis</div><div class="section-sub">Identify the strongest and weakest subject</div></div>')
        subject=pd.DataFrame({"Subject":["Mathematics","Science","English"],"Score":[math,science,english]}).set_index("Subject")
        st.bar_chart(subject,height=280)
        weakest=subject["Score"].idxmin(); strongest=subject["Score"].idxmax()
        st.info(f"Strongest subject: **{strongest} ({int(subject.loc[strongest,'Score'])})**  |  Weakest subject: **{weakest} ({int(subject.loc[weakest,'Score'])})**")

        st.html('<div class="section"><div class="section-title">📈 Performance Trend</div><div class="section-sub">Previous academic performance → internal assessment → predicted final performance</div></div>')
        trend=pd.DataFrame({"Stage":["Previous Score","Internal Marks","Predicted Exam"],"Score":[previous,internal,pred]}).set_index("Stage")
        st.line_chart(trend,height=300)

        st.html('<div class="section"><div class="section-title">💡 Personalized Recommendations</div></div>')
        for x in recommendations(student):
            st.html(f'<div class="rec">✦ &nbsp; {x}</div>')

elif page=="🏠 Dashboard":
    st.html('<div class="hero"><div class="badge">✦ ACADEMIC INTELLIGENCE</div><h1>Student <span class="grad">Performance</span> Dashboard</h1><p>Academic analytics, subject performance, student comparison and machine learning insights.</p><div class="hero-icon">🎓</div></div>')
    cols=st.columns(5)
    vals=[("👥","Students",f"{len(df):,}"),("🏆","Average Score",f"{df.Exam_Score.mean():.1f}"),("🎯","Attendance",f"{df.Attendance.mean():.1f}%"),("📚","Study Hours",f"{df.Hours_Studied.mean():.1f}"),("📝","Internal Avg",f"{df.Internal_Marks.mean():.1f}")]
    for col,(i,l,v) in zip(cols,vals):
        with col: st.html(f'<div class="metric"><div class="metric-icon">{i}</div><div class="metric-label">{l}</div><div class="metric-value">{v}</div></div>')
    st.html('<div class="section"><div class="section-title">📖 Subject Performance</div></div>')
    st.bar_chart(df[["Mathematics_Score","Science_Score","English_Score"]].mean(),height=300)
    st.html('<div class="section"><div class="section-title">📈 Overall Performance Distribution</div></div>')
    st.bar_chart(df.Exam_Score.value_counts().sort_index(),height=320)

elif page=="📊 Analytics":
    st.html('<div class="hero"><div class="badge">✦ EXPLORATORY DATA ANALYSIS</div><h1>Student <span class="grad">Analytics</span></h1><p>Explore academic factors, subject performance, participation and performance trends.</p><div class="hero-icon">📊</div></div>')
    t1,t2,t3,t4=st.tabs(["📚 Academic Factors","📖 Subjects","🧠 Learning Factors","🔥 Correlations"])
    with t1:
        c1,c2=st.columns(2)
        with c1: st.subheader("Attendance vs Exam Score"); st.scatter_chart(df,x="Attendance",y="Exam_Score",height=330)
        with c2: st.subheader("Study Hours vs Exam Score"); st.scatter_chart(df,x="Hours_Studied",y="Exam_Score",height=330)
        st.subheader("Previous Scores vs Exam Score"); st.scatter_chart(df,x="Previous_Scores",y="Exam_Score",height=330)
        st.subheader("Internal Marks vs Exam Score"); st.scatter_chart(df,x="Internal_Marks",y="Exam_Score",height=330)
    with t2:
        st.subheader("Average Subject Scores")
        st.bar_chart(df[["Mathematics_Score","Science_Score","English_Score"]].mean(),height=330)
        st.subheader("Subject Score Distribution")
        st.line_chart(df[["Mathematics_Score","Science_Score","English_Score"]].mean(axis=0),height=300)
    with t3:
        c1,c2=st.columns(2)
        with c1:
            st.subheader("Motivation vs Score")
            st.bar_chart(df.groupby("Motivation_Level")["Exam_Score"].mean().sort_values(),height=300)
        with c2:
            st.subheader("Participation vs Score")
            bins=pd.cut(df["Participation"],bins=[0,50,70,85,100],labels=["Low","Moderate","High","Very High"])
            st.bar_chart(df.assign(P=bins).groupby("P",observed=False)["Exam_Score"].mean(),height=300)
        st.subheader("Tutoring Sessions vs Score"); st.line_chart(df.groupby("Tutoring_Sessions")["Exam_Score"].mean(),height=300)
    with t4:
        numerical=df.select_dtypes(include=np.number)
        st.subheader("Numeric Factors Associated With Exam Score")
        st.bar_chart(numerical.corr()["Exam_Score"].drop("Exam_Score").sort_values(ascending=False),height=500)

else:
    st.html('<div class="hero"><div class="badge">✦ MACHINE LEARNING</div><h1>Model <span class="grad">Performance</span></h1><p>Evaluation of the regression models used for student exam-score prediction.</p><div class="hero-icon">🤖</div></div>')
    if RESULTS_PATH.exists():
        results=pd.read_csv(RESULTS_PATH)
        results.columns=["Model","MAE","RMSE","R2"]
        st.dataframe(results.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.3f}"}),use_container_width=True,hide_index=True)
        st.bar_chart(results.set_index("Model")[["RMSE","R2"]],height=350)
        best=results.sort_values("RMSE").iloc[0]
        st.info(f"Deployed model: **{best['Model']}** based on the lowest test RMSE in the retrained comparison.")
    else:
        st.warning("Run upgrade_project.py first.")

st.html('<div style="text-align:center;color:#52525b;padding:35px;font-size:.75rem">✦ EduPredict • Student Performance Prediction System • Machine Learning Capstone Project</div>')
