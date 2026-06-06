import sys
import streamlit as st

def sanitize_input(raw_input: str) -> str:
    """PHASE 1: INPUT SANITIZATION"""
    return raw_input.lower().strip()


def get_response(clean_input: str) -> str:
    """PHASE 2: PROCESS (The Deterministic Logic Engine)"""
    if "hello" in clean_input or "hi" in clean_input or "hey" in clean_input:
        return "Hello there! How can I assist you today?"
        
    elif "help" in clean_input or "commands" in clean_input:
        return "Available tracks: You can greet me, ask 'how are you', ask for the 'time', or check my 'identity'."
        
    elif "how are you" in clean_input:
        return "As a logic engine, I don't have feelings, but my processes are fully operational!"
        
    elif "identity" in clean_input or "who are you" in clean_input:
        return "I am a Rule-Based Assistant built using your presentation blueprints."
        
    elif "time" in clean_input:
        return "Time is relative, but my internal clock indicates that my execution loop is rendering perfectly."
        
    else:
        return "Command not recognized. This prompt is outside my predefined rules. Type 'help' for options!"

def run_web_interface():
    st.set_page_config(page_title="Rule-Based AI Engine", page_icon="🤖", layout="centered")
    st.title("Rule-Based AI Engine")
    st.caption("A Deterministic Guardrail 'White Box' Chat Interface")
    st.write("---")

    # Initialize chat memory on the webpage
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am a Rule-Based Logic Engine. How can I assist you today?"}
        ]

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle typing into the chat box
    if user_query := st.chat_input("Message your AI engine..."):
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Process the input using our rules
        clean_query = sanitize_input(user_query)
        bot_reply = get_response(clean_query)
        
        with st.chat_message("assistant"):
            st.write(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

def run_terminal_interface():
    print("--- Rule-Based AI Chatbot Online (Terminal Mode) ---")
    print("Type your prompt below. Type 'exit' to turn off the chatbot.\n")
    
    while True:
        raw_feed = input("You: ")
        clean_input = sanitize_input(raw_feed)
        
        if clean_input == "exit":
            print("Bot: Terminating operations. Goodbye!")
            break
            
        response = get_response(clean_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    if st.runtime.exists():
        run_web_interface()
    else:
        run_terminal_interface()