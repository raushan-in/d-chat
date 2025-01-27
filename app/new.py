import streamlit as st

# style_logo="""
# <div class="logo">
#     <img src="D://Projects/Query Queen/qq/web/static/logo.png" alt="logo Image">
# </div>
# <style>
#     .logo img {
#         width:100px;
#         heigh:100px;
#         justify-content: center;
#     }
    
# </style>
# """
#st.components.v1.html(style_logo)
#st.markdown(f'<img src="{"D://Projects/Query Queen/qq/web/static/logo.png"}" style="{style_logo}">',unsafe_allow_html=True)

# st.markdown(
#         '<img src="./web/static/logo.jpg" height="300" style="border: 5px solid orange">',
#         unsafe_allow_html=True,
#     )
col1, col2 = st.columns([4,20])
with col1:
    st.image("./web/static/logo.png", width=80)
with col2:
    st.markdown("<h1 style='text-align: center; color: Black;'>Welcome to Query Queen UI</h1>", unsafe_allow_html=True)


#st.markdown('<img src="D://Projects/Query Queen/qq/web/static/logo.png" alt="QQ" width="500" height="600">',unsafe_allow_html=True)


file = st.file_uploader("Choose a PDF file only",
                        type=["pdf"])

if file is not None:
        
    st.write(f'You uploaded {file.name}')

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

    response = f"QQ: {prompt}"

    #display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    #add assistant response to chat history
    st.session_state.messages.append({"role":"assistant", "content":"response"})


#messages = st.container(border=True,height=600)