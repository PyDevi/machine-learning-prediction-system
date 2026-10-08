import streamlit as st
st.title("Hello Chai App")
st.subheader("Brewed with streamlit")
st.text("Welcome to your first intreactive app")
st.write("Choose your favoritr variety of chai")
chai = st.selectbox("Your favorite chai:",["Masala chai","Lemon tea","Ginger tea","Kesar tea","Adrak tea"])
st.write(f"You choose {chai} . Excellent choice.")
st.success("Your chai has been brewed")