import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Supported languages and model mapp
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese": "zh"
}

@st.cache_resource
def load_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# UI
st.title("🌐 AI-Powered Multi-Language Translator")
st.markdown("Translate text between different languages using Hugging Face NLP models.")

text_input = st.text_area("Enter text to translate:", "")

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source Language", list(LANGUAGE_CODES.keys()), index=0)
with col2:
    tgt_lang = st.selectbox("Target Language", list(LANGUAGE_CODES.keys()), index=1)

if st.button("Translate"):
    if src_lang == tgt_lang:
        st.warning("Source and target languages are the same.")
    elif not text_input.strip():
        st.error("Please enter some text.")
    else:
        try:
            src_code = LANGUAGE_CODES[src_lang]
            tgt_code = LANGUAGE_CODES[tgt_lang]
            tokenizer, model = load_model(src_code, tgt_code)
            result = translate_text(text_input, tokenizer, model)
            st.success("Translation Complete:")
            st.text_area("Translated Text", result, height=150)
        except Exception as e:
            st.error(f"Error: {str(e)}")

            #streamlit run multi_lang_translator.py (for run this code)