import os
from openai import OpenAI
from duckduckgo_search import DDGS

USE_CASE = """You are a smart planning agent. When given 
a goal, break it into simple subtasks with priority 
(High/Medium/Low) and time estimate for each task."""

def get_client(api_key, base_url):
    """Initialize the OpenAI client with dynamic base URL (for Grok/Groq)."""
    return OpenAI(api_key=api_key, base_url=base_url)

def decompose_goal(goal, api_key, base_url, model):
    """Asks the LLM to break down a main goal into subtasks."""
    try:
        client = get_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": USE_CASE},
                {"role": "user", "content": f"Goal: {goal}\n\nPlease format the output exactly like this:\nTASK 1: [task name] | PRIORITY: [priority] | TIME: [time estimate]\nTASK 2: [task name] | ..."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def search_web(query):
    """DuckDuckGo web search tool."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search Error: {e}"

REACT_SYSTEM_PROMPT = """You are an autonomous execution agent. You can execute tasks using the ReAct framework: Reason, Act, Observe.
You have access to the following tool:
- search_web: "Use this to search the internet. Input should be a search query."

Use the following format strictly:
Thought: you should always think about what to do
Action: the action to take, should be one of [search_web]. If you want to use the tool, just output the tool name. If no tool is needed or you are done, output "None".
Action Input: the input to the action (the search query). If Action is "None", Action Input should be your Final Answer.
Observation: the result of the action (provided by the user/system, do not generate this yourself)
... (this Thought/Action/Action Input/Observation can repeat N times)

Example of ending the loop:
Thought: I now have enough information to answer.
Action: None
Action Input: To complete this task, you need to... [Your final detailed step-by-step answer here]
"""

def execute_task_react(task, api_key, base_url, model):
    """
    Generator function that yields the steps of the ReAct loop.
    This allows the Streamlit UI to show the agent's thought process live.
    """
    client = get_client(api_key, base_url)
    
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Execute this task:\n{task}"}
    ]
    
    max_loops = 5
    for i in range(max_loops):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stop=["Observation:"] # We stop generation so the system can provide the observation
            )
            reply = response.choices[0].message.content.strip()
            
            # Send the LLM's thought and planned action to the UI
            yield {"type": "llm_reply", "content": reply}
            
            # Add the reply to the conversation history
            messages.append({"role": "assistant", "content": reply})
            
            # Parse the Action and Action Input
            action_line = [line for line in reply.split('\n') if line.lower().startswith("action:")]
            input_line = [line for line in reply.split('\n') if line.lower().startswith("action input:")]
            
            action = "None"
            action_input = reply # Fallback if formatting is weird
            
            if action_line:
                action = action_line[-1].split(":", 1)[1].strip()
            if input_line:
                action_input = input_line[-1].split(":", 1)[1].strip()
            
            # If Action is None, we are done
            if action.lower() == "none" or not action or "search_web" not in action:
                yield {"type": "final_answer", "content": action_input}
                break
                
            # If Action is search_web, execute the tool
            if "search_web" in action:
                yield {"type": "tool_call", "action": action, "input": action_input}
                observation = search_web(action_input)
                yield {"type": "observation", "content": observation}
                
                # Append the tool's result to the conversation so the LLM can see it in the next loop
                messages.append({"role": "user", "content": f"Observation: {observation}\nNow continue based on the format."})
                
        except Exception as e:
            yield {"type": "error", "content": str(e)}
            break
    else:
        yield {"type": "error", "content": "Max reasoning loops (5) reached without finding a final answer."}

def adapt_plan(goal, feedback, api_key, base_url, model):
    """Modifies the plan based on user feedback."""
    try:
        client = get_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Original goal: {goal}\nUser feedback: {feedback}\n\nPlease create an improved plan based on this feedback."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
