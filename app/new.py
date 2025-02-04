import streamlit as st
import requests

col1, col2 = st.columns([4,20])
with col1:
    st.image("./web/static/logo.png", width=80)
with col2:
    st.markdown("<h1 style='text-align: center; color: Black;'>Welcome to Query Queen UI</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose a PDF file only", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner("Uploading... Please wait."):
            
            files = {"files": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            
            try:
                response = requests.post("http://localhost:8000/upload", files=files)  # Change URL as needed
                
                if response.status_code == 200:
                    st.success("File uploaded successfully!")
                else:
                    st.error(f"Upload failed: {response.json().get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Unexpected error: {e}")









# with st.form("Upload-form",clear_on_submit=True):
#     file = st.file_uploader("Choose a PDF file only",
#                         type=["pdf"])
#     submitted=st.form_submit_button("Submit")

#     if file is not None:
        
#         st.write(f'You uploaded {file.name}')

# if uploaded_files:
#     file_names = [file.name for file in uploaded_files] 


#file_names = st.json.load(uploaded_files)
if "messages" not in st.session_state:
    st.session_state.messages=[]

#display chat messages from history on app return
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#react to user input
if prompt := st.chat_input("How may I help you?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt.strip() and uploaded_files:
        try:
            url = "http://localhost:8000/ask"  # Replace with your API endpoint
            file_names = [file.name for file in uploaded_files] if uploaded_files else []
            payload = {
                "query": {
                    "question": prompt,       # User's question
                    "file_name": file_names  # Replace with the actual file name being queried
                }
            }            
            #payload = {"query": prompt, "document_id": file_names}  # Replace with dynamic document ID if needed
            headers = {"Content-Type": "application/json"}
        
            response = requests.post(url, json=payload, headers=headers)
        
            if response.status_code == 200:
                # Get the assistant's response from the API
                api_response = response.json().get("answer", "No response from API.")  # Assuming the API returns an 'answer' field
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": api_response})
                
                # Display assistant response in chat
                with st.chat_message("assistant"):
                    st.markdown(api_response)
            else:
                error_message = f"Error: {response.status_code}, {response.text}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown(error_message)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)







    # response = f"QQ: {prompt}"

    # #display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response)

    # #add assistant response to chat history
    # st.session_state.messages.append({"role":"assistant", "content":"response"})


#messages = st.container(border=True,height=600)