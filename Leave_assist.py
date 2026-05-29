#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------------------------------------------
# Cell 0.1 -- Install all dependencies with pinned minimum versions
# Safe to re-run. All packages are compatible with Python 3.10.
# This is to install all dependencies to use the agent and other tools
# ---------------------------------------------------------------

import subprocess, sys

packages = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-core>=0.3.0",
    "langgraph>=0.2.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pyngrok>=7.0.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "langsmith>=0.1.0",
    "python-multipart>=0.0.9",
    "requests>=2.31.0",
]

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet"] + packages
)
print("All dependencies installed successfully.")


# In[13]:


import os
import nest_asyncio

from dotenv import load_dotenv

from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel

from pyngrok import ngrok

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
#from langchain.agents import create_agent

import uvicorn

# =========================================================

# FIX JUPYTER ASYNC ISSUE

# =========================================================

nest_asyncio.apply()

# =========================================================

# LOAD ENV VARIABLES

# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

# =========================================================

# MOCK TEAM DATA

# =========================================================

employees = [
{"id": 1, "name": "Ravi Kumar", "department": "Architecture"},
{"id": 2, "name": "John Carter", "department": "Engineering"},
{"id": 3, "name": "Emily Davis", "department": "HR"},
{"id": 4, "name": "Sophia Lee", "department": "Finance"},
{"id": 5, "name": "Michael Brown", "department": "Sales"},
{"id": 6, "name": "Olivia Wilson", "department": "Support"},
{"id": 7, "name": "Daniel Thomas", "department": "Security"},
{"id": 8, "name": "Emma Taylor", "department": "Operations"},
{"id": 9, "name": "James White", "department": "Marketing"},
{"id": 10, "name": "Isabella Martin", "department": "Admin"}
]

# =========================================================

# MOCK LEAVE REQUESTS

# =========================================================

leave_requests = [
{
"employee": "Ravi Kumar",
"leave_type": "Vacation",
"days": 5,
"status": "Approved"
},
{
"employee": "John Carter",
"leave_type": "Sick Leave",
"days": 2,
"status": "Pending"
},
{
"employee": "Emily Davis",
"leave_type": "Emergency",
"days": 3,
"status": "Approved"
},
{
"employee": "Sophia Lee",
"leave_type": "Vacation",
"days": 7,
"status": "Rejected"
},
{
"employee": "Michael Brown",
"leave_type": "Personal",
"days": 1,
"status": "Approved"
},
{
"employee": "Olivia Wilson",
"leave_type": "Maternity",
"days": 30,
"status": "Approved"
},
{
"employee": "Daniel Thomas",
"leave_type": "Sick Leave",
"days": 4,
"status": "Pending"
},
{
"employee": "Emma Taylor",
"leave_type": "Vacation",
"days": 10,
"status": "Approved"
},
{
"employee": "James White",
"leave_type": "Emergency",
"days": 2,
"status": "Rejected"
},
{
"employee": "Isabella Martin",
"leave_type": "Personal",
"days": 3,
"status": "Approved"
}
]

# =========================================================

# SAMPLE QUESTIONS & RESPONSES

# =========================================================

sample_questions = [
{
"question": "What is Ravi Kumar's leave status?",
"response": "Ravi Kumar's leave is Approved for 5 days."
},
{
"question": "Who has pending leave approvals?",
"response": "John Carter and Daniel Thomas have pending approvals."
},
{
"question": "Which employees have rejected leave requests?",
"response": "Sophia Lee and James White have rejected requests."
},
{
"question": "Who is on maternity leave?",
"response": "Olivia Wilson is currently on maternity leave."
},
{
"question": "How many employees applied for vacation leave?",
"response": "Three employees applied for vacation leave."
}
]

# =========================================================

# PROFESSIONAL SYSTEM PROMPT

# =========================================================

SYSTEM_PROMPT = """
You are LeaveAssist AI, a professional HR leave management assistant.

Your responsibilities:

* Help employees check leave status
* Provide concise and empathetic responses
* Maintain professionalism at all times
* Clearly explain leave approvals, pending requests, and rejections
* Never fabricate employee information
* Respond in a calm and supportive HR tone

Response Style:

* Professional
* Concise
* Empathetic
* Human-friendly
  """

# =========================================================

# TOOL DEFINITIONS

# =========================================================

@tool
def get_leave_status(employee_name: str) -> str:
    """
    Get leave approval status for an employee.
    """

    for request in leave_requests:

        if request["employee"].lower() == employee_name.lower():

            return (
                f"Employee: {request['employee']}\n"
                f"Leave Type: {request['leave_type']}\n"
                f"Days: {request['days']}\n"
                f"Status: {request['status']}"
            )

    return "No leave record found for this employee."


# =========================================================

# INITIALIZE LLM

# =========================================================

llm = ChatOpenAI(
model=MODEL_NAME,
temperature=0
)

# =========================================================

# CREATE LANGGRAPH REACT AGENT

# =========================================================

agent = create_react_agent(
model=llm,
tools=[get_leave_status],
prompt=SYSTEM_PROMPT
)

# =========================================================

# PYDANTIC MODELS

# =========================================================

class LeaveQuery(BaseModel):
    employee_name: str

# =========================================================

# FASTAPI APP

# =========================================================

app = FastAPI(
title="LeaveAssist AI",
description="Professional Leave Management Agent using LangGraph",
version="1.0"
)

# =========================================================

# HEALTH ENDPOINT

# =========================================================

@app.get("/health")
def health():
    return {
        "status": "alive"
    }

# =========================================================

# GET EMPLOYEES

# =========================================================

@app.get("/employees")
def get_employees():
    return employees

# =========================================================

# GET LEAVE REQUESTS

# =========================================================

@app.get("/leave-requests")
def get_leave_requests():
    return leave_requests

# =========================================================

# SAMPLE QUESTIONS

# =========================================================

@app.get("/sample-questions")
def get_sample_questions():
    return sample_questions

# =========================================================

# AGENT ENDPOINT

# =========================================================

@app.post("/leave-status")
def leave_status(query: LeaveQuery):


    response = agent.invoke(
        {
        "messages": [
            {
                "role": "user",
                "content": f"What is the leave status for {query.employee_name}?"
                }
            ]
        }
    )

    return {
        "employee_name": query.employee_name,
        "agent_response": response["messages"][-1].content
    }


# In[15]:




