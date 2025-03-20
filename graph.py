import datetime
import os
from dataclasses import dataclass
from typing import Union, List

import requests
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

if not os.environ.get("OPENAI_API_KEY"):
    api_key = input("Enter your OpenAi API Key: ")
    os.environ[
        "OPENAI_API_KEY"] = api_key


@dataclass
class AgentState:
    messages: List[BaseMessage]  # List of messages (AI & Human)


# Define a simple tool
def my_info(_):
    return {
        "user_details": {
            "name": "Amit Chaudhary",
            "designation": "Sr. Mobile Application Developer",
            "description": "A passionate Mobile App Developer with over 5 years of experience in creating engaging, high-performance cross-platform mobile applications. Adept in languages including Java, Kotlin, Dart, JavaScript and Python, and highly eager to contribute to your team.",
            "picture": {
                "src": "amit.jpeg",
                "link": "https://www.linkedin.com/in/itheamc"
            },
            "startDate": "01 Apr 2019",
            "updateDate": "19 Mar 2025",
            "links": [
                {
                    "icon": "fa fa-envelope-open",
                    "tooltip": "Send Mail",
                    "label": "itheamc@gmail.com",
                    "link": "mailto:itheamc@gmail.com?subject=Job%20offer"
                },
                {
                    "icon": "fa fa-map-marker-alt",
                    "tooltip": "View in maps",
                    "label": "Dang, Nepal",
                    "link": "https://maps.app.goo.gl/GcQPye3irj7LveNQ7"
                },
                {
                    "icon": "fa fa-mobile-alt",
                    "tooltip": "Call",
                    "label": "+977-9847967132",
                    "link": "tel:+9779847967132"
                }
            ],
            "sns": [
                {
                    "icon": "fab fa-github",
                    "tooltip": "Github",
                    "link": "https://github.com/itheamc"
                },
                {
                    "icon": "fab fa-stack-overflow",
                    "tooltip": "Stack Overflow",
                    "link": "https://stackoverflow.com/users/16758002/itheamc"
                },
                {
                    "icon": "fab fa-linkedin",
                    "tooltip": "LinkedIn",
                    "link": "https://www.linkedin.com/in/itheamc/"
                }
            ],
            "qrCode": "qr-code.png"
        },

        "skills": [
            {
                "icon": "fab fa-android",
                "title": "Android",
                "scale": 5,
                "tech": ["Java", "Kotlin", "Jetpack Compose", "Room", "Gradle"],
                "lib": ["RxJava", "LiveData", "Retrofit", "Firebase"]
            },
            {
                "icon": "<div class='flutter-icon'></div>",
                "title": "Flutter",
                "scale": 5,
                "tech": ["Dart", "pub", "Riverpod", "Provider"],
                "lib": ["BLoC", "GetX", "Firebase", "FCM"]
            },
            {
                "icon": "fab fa-apple",
                "title": "iOS",
                "scale": 3,
                "tech": ["Swift", "Cocoapod"],
                "lib": ["SQLite.swift", "Firebase"]
            },
            {
                "icon": "fa fa-globe",
                "title": "Web",
                "scale": 3,
                "tech": ["HTML", "CSS", "JavaScript", "TypeScript"],
                "lib": ["jQuery", "Bootstrap"]
            },
            {
                "icon": "fab fa-react",
                "title": "React.JS",
                "scale": 3,
                "tech": ["JavaScript", "TypeScript", "npm"],
                "lib": ["Redux", "FCM"]
            },
            {
                "icon": "fab fa-python",
                "title": "Python",
                "scale": 3,
                "tech": ["Python", "Django", "pip"],
                "lib": ["flet", "chaquopy", "OpenCV", "geodjango"]
            }
        ],

        "languages": [
            {"icon": "&#x0905;", "name": "Nepali", "scale": 5, "proficiency": "Native or Bilingual Proficiency"},
            {"icon": "&#x0905;", "name": "Hindi", "scale": 4, "proficiency": "Near-Native Proficiency"},
            {"icon": "A", "name": "English", "scale": 3, "proficiency": "Professional Working Proficiency"}
        ],

        "interests": [
            {"icon": "fas fa-book", "title": "Reading"},
            {"icon": "fas fa-search", "title": "Researching"},
            {"icon": "fas fa-code", "title": "Coding"},
            {"icon": "fas fa-plane-departure", "title": "Travelling"}
        ],

        "personal": [
            {"icon": "fa fa-language", "label": "Nationality", "value": "Nepali"},
            {"icon": "fa fa-heart", "label": "Marital Status", "value": "Married"}
        ],

        "experiences": [
            {
                "position": "Sr. Mobile Application Developer",
                "company": "NAXA Pvt. Ltd, Kathmandu",
                "duration": "July 2022 - Present",
                "tech": ["Android (Native)", "Flutter", "iOS (Swift)"],
                "achievements": [
                    "Lead the development and maintenance of mobile applications for iOS and Android platforms.",
                    "Collaborate with cross-functional teams to define and implement new features.",
                    "Optimize app performance and troubleshoot issues."
                ]
            },
            {
                "position": "Flutter Developer",
                "company": "Casper India, Bangalore",
                "duration": "Jan 2022 - May 2022",
                "tech": ["Flutter", "Python", "React.JS", "Django"],
                "achievements": [
                    "Developed robust, location-specific ecommerce apps utilizing Flutter.",
                    "Used Django framework to build a backend billing application."
                ]
            }
        ],

        "education": [
            {
                "board": "Pokhara University",
                "school": "Victoria College/Dang, Nepal",
                "concentration": "Bachelor of Business Administration",
                "score": 3.6,
                "metric": "CGPA",
                "duration": "July 2011 - Sept 2015"
            },
            {
                "board": "Higher Secondary Education Board (HSEB)",
                "school": "Janta Higher Secondary School/Gadhawa, Nepal",
                "concentration": "10+2 (Commerce)",
                "score": 53,
                "metric": "%",
                "duration": "Apr 2009 - Mar 2011"
            }
        ],

        "personal_projects": [
            {
                "name": "naxalibre",
                "description": "Feature-rich MapLibre plugin for Flutter.",
                "duration": "Feb 2025",
                "tech": ["Flutter", "Dart", "Pigeon", "Kotlin", "Swift", "MapLibre"],
                "refs": [
                    {"icon": "fa fa-link", "tooltip": "Check it out", "url": "https://pub.dev/packages/naxalibre/"}]
            }
        ],

        "company_projects": [
            {
                "name": "Starter Template (Flutter)",
                "description": "Streamline flutter project setup.",
                "duration": "Dec 2024 - Mar 2025",
                "tech": ["Flutter", "Riverpod", "Go Router", "Dio", "FCM", "Location", "In App Update"],
                "refs": [
                    {"icon": "fa fa-link", "tooltip": "Visit", "url": "https://github.com/itheamc/starter-template"}]
            }
        ]

    }


def hello_tool(name: str):
    return f"Hello, {name}! What's up?"


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


def fetch_todos(_) -> Union[list[dict], None]:
    try:
        res = requests.get("https://jsonplaceholder.typicode.com/todos")
        res.raise_for_status()
        return res.json()
    except requests.RequestException or requests.JSONDecodeError:
        return []


def fetch_todo_by_id(todo_id: int) -> Union[dict, None]:
    try:
        res = requests.get(f"https://jsonplaceholder.typicode.com/todos/{todo_id}")
        res.raise_for_status()
        return res.json()
    except requests.RequestException or requests.JSONDecodeError:
        return {}


def current_weather_of(city: str) -> Union[list[dict], dict, None]:
    try:
        res = requests.get(f"https://api.weatherapi.com/v1/current.json?key=74bfb20dcecc4c7792035545251903&q={city}")
        res.raise_for_status()
        return res.json()
    except requests.RequestException or requests.JSONDecodeError:
        return []


info = Tool(
    name="GetMyInfo",
    func=my_info,
    description="Get information about me"
)

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

todos = Tool(
    name="FetchTodos",
    func=fetch_todos,
    description="Fetch the todos. It doesn't take any input."
)

todo = Tool(
    name="FetchTodoByid",
    func=fetch_todo_by_id,
    description="Fetch the todo by id. It take int id as a input."
)

weathers = Tool(
    name="FetchCurrentWeather",
    func=current_weather_of,
    description="Fetch current weather information of the given city"
)

# Initialize the agent
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define the agent
# Track messages in state
graph = StateGraph(AgentState)

# Use LangGraph’s built-in ReAct agent
graph.add_node("agent", create_react_agent(model, [info, hello, date_time, user, users, todos, todo, weathers]))

# Define graph flow
graph.set_entry_point("agent")

# Compile the graph
agent_executor = graph.compile()

# Maintain conversation history
conversation_history = []


# Extracting AI Response
def extract_ai_response(response_dict):
    # Get the messages list
    messages = response_dict.get('messages', [])

    # Find the last AIMessage
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return message.content

    # If no AIMessage with content is found
    return None


# Example interaction
def chat(user_input):
    global conversation_history

    # Append user message to history
    conversation_history.append(HumanMessage(content=user_input))

    # Prepare input with history
    prompts = {"messages": conversation_history}

    # Invoke agent
    result = agent_executor.invoke(prompts)

    # Extract AI response
    response = extract_ai_response(result)

    # Append AI response to history
    if response:
        conversation_history.append(AIMessage(content=response))

    return response


if __name__ == "__main__":
    while True:
        prompt = input("Prompt: ")
        ai_response = chat(prompt)
        print(ai_response)
