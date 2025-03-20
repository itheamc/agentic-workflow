import datetime
import os
from typing import Union

import requests
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import init_chat_model
from langchain.tools import Tool

if not os.environ.get("OPENAI_API_KEY"):
    api_key = input("Enter your OpenAi API Key: ")
    os.environ[
        "OPENAI_API_KEY"] = api_key


# Define a simple tool
def hello_tool(name: str):
    return f"Hello, {name}!"


def current_date_time(_):
    return f"{datetime.datetime.now()}"


def get_user_by_id(user_id: int) -> Union[dict, None]:
    try:
        res = requests.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
        res.raise_for_status()
        return res.json()
    except requests.RequestException or requests.JSONDecodeError:
        return {}


def fetch_users(_) -> Union[list[dict], None]:
    try:
        res = requests.get("https://jsonplaceholder.typicode.com/users")
        res.raise_for_status()
        return res.json()
    except requests.RequestException or requests.JSONDecodeError:
        return []


hello = Tool(
    name="GreetUser",
    func=hello_tool,
    description="Greets the user by name"
)

date_time = Tool(
    name="GetCurrentDateTime",
    func=current_date_time,
    description="Get the current date and time data"
)

user = Tool(
    name="GetUser",
    func=get_user_by_id,
    description="Get user by given id"
)

users = Tool(
    name="FetchUsers",
    func=fetch_users,
    description="Fetch the users. It doesn't take any input."
)

# Initialize the agent
model = init_chat_model("gpt-4o-mini", model_provider="openai")

agent = initialize_agent(
    tools=[hello, date_time, user, users],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Test the agent
# response = agent.invoke("Say hello to Norton")
# print(response)

# Test the for users
response = agent.invoke("Can you list the email of users?")
print(response)
