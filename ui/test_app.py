import streamlit as st
st.set_page_config(layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); padding: 2rem; border-radius: 12px;">
    <h1 style="color: white; margin: 0;">Test Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.write("Does this render?")
col1, col2 = st.columns(2)
with col1:
    st.info("Left column")
with col2:
    st.success("Right column")
