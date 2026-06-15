from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """You are an expert software engineer conducting a rigorous code review. 
    Analyze the provided code snippet and return your evaluation strictly using the structure below.

    ## 1. Quality Scores (0-10)
    * **Maintainability:** [Score]/10 - (How easy is it to update and scale?)
    * **Readability:** [Score]/10 - (Are variables named well? Is the logic easy to follow?)
    * **Performance/Efficiency:** [Score]/10 - (Are there redundant operations or slow loops?)

    ## 2. Executive Summary
    [Provide 1-2 concise sentences summarizing the overall quality and the primary takeaway.]

    ## 3. Critical Issues & Bugs
    * [Issue 1: Describe the issue, why it is a problem, and the severity.]
    * [Issue 2: ...]
    *(If no critical issues exist, output "None detected.")*

    ## 4. Best Practices & Improvements
    * [Improvement 1: Provide actionable advice on how to write this cleaner or more idiomatically.]
    * [Improvement 2: ...]

    ## 5. Refactored Snippet
    [Provide a short, refactored version of the code that implements your suggestions.]
"""


checkpointer = InMemorySaver()


model = init_chat_model(
    model = "gemini-2.5-flash-lite",
    model_provider="google-genai",
    temperature=0.5,
    timeout=600,
    max_tokens=25000,
    streaming=True,
)

content = """
def calculate_factorial(n):
    if n == 0: 
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

"""

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "review-session-1"}}


response = agent.invoke({
    "messages":[
        {
            "role": "user",
            "content": content,
        }
    ]
}, config=config)

print(response["messages"][-1].content_blocks)