import streamlit as st
from streamlit_option_menu import option_menu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

file_path = r"..."
df = pd.read_csv(file_path)

def random_question(file_path):
    if not os.path.exists(file_path):
        st.error("File dataset tidak ditemukan. Pastikan path benar.")
        return None, None, None, None, None 

    df = pd.read_csv(file_path)

    if df.empty:
        st.error("Dataset kosong.")
        return None, None, None, None, None  

    random_row = df.sample(n=1).iloc[0]

    return (
        random_row["text"],
        random_row["target"],
        random_row["answer"],
        random_row.get("style", "default"),  
        random_row.get("category", "default")  
    )  

def get_quiz_data(file_path):
    if st.session_state.get("new_question", False):
        paragraf_info, pertanyaan, jawaban_benar, style, category = random_question(file_path)    

        if all([paragraf_info, pertanyaan, jawaban_benar]):  
            st.session_state.quiz_data = {
                "paragraf_info": paragraf_info,
                "pertanyaan": pertanyaan,
                "jawaban_benar": jawaban_benar,
                "style": style if style else "default",  
                "category": category if category else "default",  
                "feedback": ""
            }

        st.session_state.new_question = False  
        return st.session_state.quiz_data  

    if "latest_question" in st.session_state:
        st.session_state.quiz_data = st.session_state.latest_question  
        st.session_state.quiz_data["feedback"] = ""  

        st.session_state.quiz_data.setdefault("style", "default")  
        st.session_state.quiz_data.setdefault("category", "default")  

        del st.session_state.latest_question  

        return st.session_state.quiz_data 

    if "quiz_data" not in st.session_state:
        paragraf_info, pertanyaan, jawaban_benar, style, category = random_question(file_path)    

        if all([paragraf_info, pertanyaan, jawaban_benar]):  
            st.session_state.quiz_data = {
                "paragraf_info": paragraf_info,
                "pertanyaan": pertanyaan,
                "jawaban_benar": jawaban_benar,
                "style": style if style else "default",  
                "category": category if category else "default",  
                "feedback": ""
            }

    return st.session_state.quiz_data 

def answer_result(user_answer, quiz_data, semantic_model):
    if not user_answer.strip():
        st.warning("Harap isi jawaban terlebih dahulu!")
        return
    
    correct_answer = quiz_data["jawaban_benar"]

    user_embedding = semantic_model.encode(user_answer, convert_to_tensor=True)
    correct_embedding = semantic_model.encode(correct_answer, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(user_embedding, correct_embedding).item()

    threshold = 0.75
    if similarity_score >= threshold:
        feedback = "✅ Jawaban Anda benar!"
        st.session_state.show_correct_answer = False  
    else:
        feedback = "❌ Jawaban Anda salah atau kurang lengkap."
        st.session_state.show_correct_answer = True 

    st.session_state.quiz_data["feedback"] = feedback
    st.session_state.quiz_data["correct_answer"] = correct_answer

@st.cache_resource
def load_model():
    model_path = r"..."
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    question_generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  
    )

    semantic_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    return question_generator, semantic_model

question_generator, semantic_model = load_model()

def generate_question(text, answer, category, style, **kwargs):
    prompt = f"""context: {text}
answer: {answer}
category: {category}
style: {style}"""

    question = question_generator(prompt, **kwargs)[0]["generated_text"]
    return question

def check_input(text, answer, style, category):
    errors = []
    
    if not text:
        errors.append("Silahkan masukkan paragraf informasi.")
    elif len(text.split()) < 50:
        errors.append(f"Paragraf informasi harus memiliki minimal 50 kata. Saat ini hanya ada {len(text.split())} kata.")
    
    if not answer:
        errors.append("Silahkan masukkan jawaban.")
    
    if style is None:
        errors.append("Silakan pilih jenis kata tanya.")
    
    if category is None:
        errors.append("Silakan pilih kategori pertanyaan.")
    
    return errors

@st.dialog("Pertanyaan yang Dihasilkan")
def pop_question(question, text, answer, style, category):
    st.write(question)
    
    if st.button("Ubah menjadi kuis"):
        save_data(text, question, answer, style, category, user_answer=None)

def save_data(text, question, answer, style, category, user_answer):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["text", "target", "answer", "style", "category"])
    
    new_data = pd.DataFrame([{
        "text": text,
        "target": question,
        "answer": answer,
        "style": style,
        "category": category
    }])

    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(file_path, index=False)

    st.session_state.latest_question = {
        "paragraf_info": text,
        "pertanyaan": question,
        "jawaban_benar": answer,
        "style": style,  
        "category": category  
    }

    st.success("Data berhasil diubah menjadi kuis dan ditampilkan di halaman kuis!")
    st.rerun() 

def gui_kuis(file_path): 
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        st.title("Kuis Pertanyaan")

    quiz_data = get_quiz_data(file_path)
    if quiz_data is None:
        return  

    with st.container():
        st.info(quiz_data["paragraf_info"])  
        st.write("##### Ayo jawab pertanyaan berikut!")
        st.info(quiz_data["pertanyaan"])   
        
    with st.form(key="quiz_form"):
        user_answer = st.text_input("Jawaban Anda", "")
        submit = st.form_submit_button("Jawab pertanyaan")  
        
        if submit:
            answer_result(user_answer, quiz_data, semantic_model)
            
    if "feedback" in st.session_state.quiz_data:
        st.write(st.session_state.quiz_data["feedback"])

    if st.session_state.get("show_correct_answer", False):
        if st.button("Tampilkan Jawaban Benar"):
            st.info(f"Jawaban yang benar adalah {st.session_state.quiz_data['correct_answer']}")
    
    cold, cole = st.columns(2)
    with cold:
        if st.button("Soal Baru"):
            st.session_state.new_question = True  
            st.session_state.quiz_data["feedback"] = ""  
            st.session_state.show_correct_answer = False  
            st.rerun() 
    
def gui_buat():
    colx, coly, colz = st.columns([2, 2, 2]) 
    with coly:
        st.title("Buat Soal")

    with st.form(key="input_form"):
        text = st.text_area(
            "Paragraf Informasi (Minimal 50 kata)", 
            placeholder="Masukkan paragraf informasi"
            )
        
        answer = st.text_input(
            "Jawaban",
            placeholder="Masukkan jawaban"
            )

        style = st.selectbox(
            "Jenis Kata Tanya", 
            ["Apa", "Kapan", "Dimana", "Berapa", "Siapa", "Bagaimana"], 
            index=None, 
            placeholder="Pilih jenis kata tanya"
            )
        
        category = st.selectbox(
            "Kategori Pertanyaan", 
            ["Sejarah", "Geografis", "Keagamaan", "Bangunan"], 
            index=None, 
            placeholder="Pilih kategori pertanyaan"
            )

        submit = st.form_submit_button("Buat Soal")

    if submit:
        errors = check_input(text, answer, style, category)

        if errors:
            for error in errors:
                st.warning(error)
        else:
            generated_question = generate_question(text, answer, category, style, max_length=128, num_beams=4, temperature=0.7)
            st.subheader("Ini pertanyaannya!")
            st.write(generated_question)
            
            pop_question(generated_question, text, answer, style, category)

selected = option_menu(
    menu_title=None,
    options=["Kuis", "Buat Soal"],
    icons=["house", "folder"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "black", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#ddd",
        },
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

if selected == "Kuis":
    gui_kuis(file_path)
elif selected == "Buat Soal":
    gui_buat()