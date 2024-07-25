css = '''
<style>
.chat-message{
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom:1rm; 
    display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
    width: 15%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
}
'''
# </style>




bot_template = '''
<div class="chat-message bot" style="margin-top: 5px; margin-bottom:5px">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNHom2uHXjuKD-DJ4C_P9_CNro3EYEvlOVgw&s">
    </div>
    <div class="message"> {{MSG}}</div>
    
</div>

'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/qWBwpNb/Photo-logo-5.png">
    </div>
    <div class="message"> {{MSG}}</div>
</div>

'''
