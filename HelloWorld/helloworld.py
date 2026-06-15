from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
agent = create_agent(    
    model = "google_genai:gemini-2.5-flash-lite",
    system_prompt = "You are a helpful assistant that translates text from English to Urdu."

)
response = agent.invoke({
    "messages":[
        {
            "role": "user",
            "content": "Translate the following English text to Urdu: 'Hello, how are you?'"
        }
    ]
})
print(response["messages"][-1].content_blocks)