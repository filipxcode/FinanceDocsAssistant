import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
# --- KONFIGURACJA ---
API_URL = "http://localhost:8000"  
st.set_page_config(
    page_title="FinAI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_status():
    try:
        res = requests.get(f"{API_URL}/status", timeout=2)
        if res.status_code == 200:
            data = res.json()
            status = data.get("status")
            
            if status == "loading":
                st.warning("⏳ Serwer uruchamia modele AI... Proszę czekać, niektóre funkcje mogą być niedostępne.", icon="🚦")
                return False 
            elif status == "ok":
                return True
    except requests.exceptions.ConnectionError:
        st.error("🔴 Brak połączenia z serwerem API. Upewnij się, że backend działa.", icon="🔌")
        return False
    except Exception as e:
        st.error(f"🔴 Błąd API: {e}", icon="⚠️")
        return False
    return True

is_online = check_api_status()
if not is_online:
    st.sidebar.warning("Tryb Offline")

def api_load_history(chat_id):
    """Pobiera historię wiadomości dla wybranego czatu"""
    try:
        res = requests.get(f"{API_URL}/chats/{chat_id}")
        if res.status_code == 200:
            data = res.json()
            backend_msgs = data.get("messages", [])
            
            loaded_msgs = []
            for m in backend_msgs:
                loaded_msgs.append({
                    "role": m["role"],
                    "content": m["text"],
                    "metrics": m.get("metrics"),
                    "sources": m.get("sources")
                })
            st.session_state.messages = loaded_msgs
    except Exception as e:
        st.error(f"Nie udało się załadować historii: {e}")
        
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "job_queue" not in st.session_state:
    st.session_state.job_queue = [] 
    
if "active_chat_id" not in st.session_state:
    query_params = st.query_params
    url_chat_id = query_params.get("chat_id", None)
    st.session_state.active_chat_id = url_chat_id
    if url_chat_id:
        api_load_history(url_chat_id)
    
if "chat_list" not in st.session_state:
    st.session_state.chat_list = [] 

if "document_list" not in st.session_state:
    st.session_state.document_list = []

def api_create_chat(title: str | None = None):
    try:
        json_payload = {"title": title if title else None}
        res = requests.post(f"{API_URL}/chats", json=json_payload)
        if res.status_code == 200:
            data = res.json()
            new_id = data["id"]
            new_title = data["title"]
            st.session_state.active_chat_id = new_id
            st.session_state.messages = [] 
            st.toast(f"Utworzono nowy czat: {new_title}")
            api_refresh_chat_list() 
            return True
    except Exception as e:
        st.error(f"Błąd tworzenia czatu: {e}")
    return False

def api_refresh_chat_list():
    """Pobiera listę dostępnych czatów"""
    try:
        res = requests.get(f"{API_URL}/chats")
        if res.status_code == 200:
            st.session_state.chat_list = res.json()
    except Exception as e:
        st.error(f"Błąd pobierania listy czatów: {e}")

        

def api_delete_chat(chat_id):
    try:
        res = requests.delete(f"{API_URL}/chats/{chat_id}")
        if res.status_code == 200:
            api_refresh_chat_list()
    except Exception as e:
        st.error(f"Nie udało się usunąć czatu: {e}")
        

def api_title_update_chat(chat_id, title):
    try:
        res = requests.patch(f"{API_URL}/chats/{chat_id}", json={"title":title})
        if res.status_code == 200:
            api_refresh_chat_list()
    except Exception as e:
        st.error(f"Nie udało się zmienić tytułu: {e}")

def api_list_documents(limit: int = 20):
    try:
        res = requests.get(f"{API_URL}/documents", params={"limit":limit})
        if res.status_code == 200:
            st.session_state.document_list = res.json()
    except Exception as e:
        st.error(f"Nie udalo sie pobrac dokumentów: {e}")

def api_delete_document(doc_id):
    try:
        res = requests.delete(f"{API_URL}/documents/{doc_id}")
        if res.status_code == 200:
            st.toast("✅ Dokument został usunięty")
            return True
        else:
            st.error(f"Błąd usuwania: {res.text}")
    except Exception as e:
        st.error(f"Błąd komunikacji: {e}")
    return False
        
def update_job_statuses():
    active_jobs_exist = False
    
    for job in st.session_state.job_queue:
        if job['status'] == "processing":
            active_jobs_exist = True
            try:
                res = requests.get(f"{API_URL}/jobs/{job['id']}")
                if res.status_code == 200:
                    new_status = res.json().get("status")
                    job['status'] = new_status 
            except:
                job['status'] = "unknown"
    
    return active_jobs_exist

@st.dialog("Utwórz nową rozmowę")
def create_chat_dialog():
    title = st.text_input("Tytuł (opcjonalnie)", max_chars=100)
    if st.button("Utwórz"):
        api_create_chat(title)
        st.rerun()

@st.dialog("Zmień tytuł")
def set_title(chat_id):
    title = st.text_input("Wpisz tytuł", key=f"input_title_{chat_id}", max_chars=100)
    if st.button("Zapisz", key=f"save_title_{chat_id}"):
        api_title_update_chat(chat_id, title)
        st.rerun()

@st.dialog("📂 Zarządzanie Plikami")
def show_files():
    st.write("Przeglądaj i zarządzaj zaindeksowanymi dokumentami.")
    
    col_opt1, col_opt2 = st.columns([2, 2])
    with col_opt1:
        load_all = st.checkbox("Pokaż wszystkie (bez limitu)")
    
    limit = 10000 if load_all else 20
    
    if st.button("🔄 Odśwież listę"):
        api_list_documents(limit)
        st.rerun()

    # Auto load if empty
    if not st.session_state.document_list:
        api_list_documents(limit)
        
    docs = st.session_state.document_list
    
    if not docs:
        st.info("Brak dokumentów.")
        return

    st.markdown(f"Liczba plików: **{len(docs)}**")
    st.divider()

    # Table header
    c1, c2, c3 = st.columns([5, 3, 1])
    c1.markdown("**Plik**")
    c2.markdown("**Data**")
    c3.markdown("**Usuń**")
    
    for doc in docs:
        col1, col2, col3 = st.columns([5, 3, 1])
        with col1:
            st.write(f"📄 {doc['original_filename']}")
            st.caption(f"Rozmiar: {doc['size_bytes'] / 1024:.1f} KB")
        with col2:
            st.write(doc['created_at'][:10])
        with col3:
            if st.button("❌", key=f"del_{doc['id']}"):
                if api_delete_document(doc['id']):
                    api_list_documents(limit)
                    st.rerun()
        st.divider()
with st.sidebar:
    st.title("🎛️ Panel Sterowania")

    with st.expander("📤 Wgraj nowe pliki", expanded=True):
        uploaded_files = st.file_uploader("Wybierz PDF", type=["pdf","pptx","txt","docx"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Uruchom przetwarzanie"):
            files_payload = [("files", (f.name, f, "application/pdf")) for f in uploaded_files]
            
            with st.spinner("Wysyłanie..."):
                try:
                    res = requests.post(f"{API_URL}/upload", files=files_payload)
                    if res.status_code == 200:
                        data = res.json()
                        job_id = data.get("job_id")
                        errors = data.get("errors", [])
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.job_queue.insert(0, {
                            "id": job_id,
                            "files": ", ".join([f.name for f in uploaded_files]),
                            "status": "processing",
                            "created_at": timestamp
                        })
                        
                        if errors:
                            st.error("⚠️ Wykryto problemy z niektórymi plikami:")
                            for err in errors:
                                st.warning(err)
                        else:
                            st.success("Zadanie dodane do kolejki!")
                            time.sleep(0.5)
                            st.rerun() 
                    else:
                        st.error(f"Błąd API: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Błąd: {e}")

        st.divider()
        if st.button("📂 Zarządzaj plikami", use_container_width=True):
            show_files()
    st.divider()

    st.subheader("📋 Status Zadań")
    if st.button("🔄 Odśwież statusy"):
        update_job_statuses()
        
    if st.session_state.job_queue:
        df = pd.DataFrame(st.session_state.job_queue)
        
        def get_status_icon(status):
            if status == "completed": return "✅ Gotowe"
            if status == "processing": return "⏳ W toku"
            if status == "failed": return "❌ Błąd"
            return "❓ Nieznany"

        df['Status'] = df['status'].apply(get_status_icon)
        
        st.dataframe(
            df[['created_at', 'files', 'Status']],
            column_config={
                "created_at": "Godzina",
                "files": "Pliki",
                "Status": "Stan"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.caption("Brak zadań w tej sesji.")
    
    st.subheader("💬 Twoje Rozmowy")
    
    if st.button("➕ Nowa Rozmowa", use_container_width=True):
        create_chat_dialog()

    if not st.session_state.chat_list:
        api_refresh_chat_list()

    for chat in st.session_state.chat_list:
        label_title = chat["title"]
        if st.session_state.active_chat_id == chat["id"]:
            label_title = f"🔵 {label_title}"
        
        col1, col2, col3 = st.columns([6, 1, 2])
        with col1:
            if st.button(label_title, key=f"select_{chat['id']}", use_container_width=True):
                st.session_state.active_chat_id = chat["id"]
                st.query_params["chat_id"] = chat["id"]
                api_load_history(chat["id"])
                st.rerun()
        with col2:
            if st.button("✏️", key=f"edit_{chat['id']}", help="Zmień tytuł"):
                set_title(chat["id"])
        with col3:
            if st.button("Usuń", key=f"delete_{chat['id']}"):
                api_delete_chat(chat["id"])
                if st.session_state.active_chat_id == chat["id"]:
                    st.session_state.active_chat_id = None
                st.rerun()

st.title("📊 FinAI Assistant")

if not st.session_state.active_chat_id:
    st.info("👈 Wybierz rozmowę z menu po lewej lub utwórz nową, aby rozpocząć.")
    if not st.session_state.chat_list:
        if st.button("Rozpocznij pierwszą analizę"):
            api_create_chat()
            st.rerun()
    st.stop()

@st.dialog("Tekst źródłowy")
def show_node(node):
    st.markdown("### Treść fragmentu:")
    st.markdown(f"> {node.get('node_content', 'Brak treści')}")
    
@st.dialog("📉 Kluczowe Dane Finansowe")
def show_metrics(metrics_list):
    if not metrics_list:
        st.warning("Brak danych do wyświetlenia.")
        return

    df = pd.DataFrame(metrics_list)
    
    column_mapping = {
        "label": "Wskaźnik",
        "amount": "Wartość",
        "unit": "Jednostka",
        "currency": "Waluta",
        "date": "Okres"
    }
    
    existing_cols = [c for c in column_mapping.keys() if c in df.columns]
    df_display = df[existing_cols].rename(columns=column_mapping)

    st.dataframe(
        df_display, 
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    csv = df_display.to_csv(index=False).encode('utf-8')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="📥 Pobierz jako CSV",
            data=csv,
            file_name=f"dane_finansowe_{datetime.now().strftime('%H%M%S')}.csv",
            mime="text/csv",
        )
    with col2:
        if st.button("Zamknij"):
            st.rerun()
            
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "metrics" in message and message["metrics"]:
            if st.button("📊 Zobacz tabelę danych", key=f"hist_metrics_{i}"):
                show_metrics(message["metrics"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📚 Zobacz Źródła"):
                for idx, src in enumerate(message["sources"]):
                    unique_key = f"hist_src_{i}_{idx}"
                    pdf_link = f"{API_URL}/files/{src['filename']}#page={src['page_ref']}"
                    st.markdown(f"- [**{src['filename']} - Strona {src['page_ref']}**]({pdf_link})")
                    if st.button("🔍 Tekst", key=unique_key):
                        show_node(src)

if prompt := st.chat_input("Napisz pytanie... np. Jaki jest prognozowany wzrost PKB w 2025?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizuję raporty..."):
            try:
                payload = {
                    "query": prompt,
                    "chat_id": st.session_state.active_chat_id
                }
                
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    r_data = response.json()
                    llm_out = r_data.get("llm_output", {})
                    summary = llm_out.get("summary_text", "Brak odpowiedzi tekstowej.")
                    key_numbers = llm_out.get("key_numbers")
                    source_data = r_data.get("source_data", [])
                    
                    st.markdown(summary)
                    
                    if key_numbers:
                        if st.button("📊 Zobacz tabelę danych", key="new_metrics_btn"):
                            show_metrics(key_numbers)
                            
                    if source_data:
                        with st.expander("📚 Źródła Dokumentów"):
                            for idx, src in enumerate(source_data):
                                unique_key = f"new_src_{idx}"
                                pdf_link = f"{API_URL}/files/{src['filename']}#page={src['page_ref']}"
                                st.markdown(f"- [**{src['filename']} - Strona {src['page_ref']}**]({pdf_link})")
                                if st.button("🔍 Tekst", key=unique_key):
                                    show_node(src)
                                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": summary,
                        "metrics": key_numbers,
                        "sources": source_data
                    })
                    st.rerun()

                else:
                    error_msg = f"Błąd API: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except Exception as e:
                error_msg = f"Nie udało się połączyć z serwerem: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})