#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------------------------------------------
# Cell 0.1 -- Install all dependencies with pinned minimum versions
# Safe to re-run. All packages are compatible with Python 3.10.
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


# In[3]:


# ---------------------------------------------------------------
# Cell 0.2 -- Load environment variables and validate all required keys
# The notebook stops here with a clear message if any key is missing.
# This prevents cryptic failures 10 cells later.
# ---------------------------------------------------------------

import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

REQUIRED_KEYS = [
    "OPENAI_API_KEY",
    "LANGSMITH_PROJECT",
    "NGROK_AUTH_TOKEN",
]

# Step 1 -- Check all keys are present in .env
missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {missing}\n"
        "Please check your .env file and re-run this cell."
    )

# Step 2 -- Validate OpenAI key format and authenticity
# A real OpenAI key starts with sk- and is at least 40 characters
openai_key = os.getenv("OPENAI_API_KEY")

print("Validating OpenAI API key...")
print(f"  Key preview  : {openai_key[:12]}...")
print(f"  Key length   : {len(openai_key)} characters")

# Format check
if openai_key.startswith("voc-"):
    raise EnvironmentError(
        "Your OPENAI_API_KEY starts with 'voc-' -- this is a Vocareum key.\n"
        "Vocareum keys only work inside Vocareum and will fail on Render and Cloud Run.\n"
        "Fix: Replace with your personal OpenAI key from platform.openai.com"
    )

if not openai_key.startswith("sk-"):
    raise EnvironmentError(
        f"Your OPENAI_API_KEY does not look like a valid OpenAI key.\n"
        f"Expected it to start with 'sk-' but got: {openai_key[:10]}...\n"
        "Fix: Check your .env file and paste the correct key from platform.openai.com"
    )

if len(openai_key) < 40:
    raise EnvironmentError(
        f"Your OPENAI_API_KEY looks too short ({len(openai_key)} characters).\n"
        "It may have been copied incorrectly -- truncated or missing characters.\n"
        "Fix: Recopy the full key from platform.openai.com"
    )

# Live check -- make a real call to OpenAI to confirm the key works
print("  Testing key against OpenAI API...")
try:
    response = requests.post(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {openai_key}"},
        timeout=10
    )
    if response.status_code == 200:
        print("  ✅ OpenAI key is valid and working")
    elif response.status_code == 401:
        raise EnvironmentError(
            "OpenAI returned 401 -- key is invalid or has been revoked.\n"
            "Fix: Generate a new key at platform.openai.com/api-keys"
        )
    elif response.status_code == 429:
        print("  ⚠️  OpenAI key is valid but rate limited -- you have hit your quota")
        print("      Check your usage at platform.openai.com/usage")
    else:
        print(f"  ⚠️  OpenAI returned unexpected status: {response.status_code}")
        print("      Key may still work -- proceed with caution")
except requests.exceptions.ConnectionError:
    print("  ⚠️  Could not reach OpenAI -- check internet connection")
    print("      Skipping live validation -- proceeding anyway")

print()

# Step 3 -- Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

print("Environment validated. All required keys are present.")
print(f"LangSmith project : {os.getenv('LANGSMITH_PROJECT')}")


# In[4]:


# ---------------------------------------------------------------
# Cell 1.1 -- Validate all imports before writing application code
# If any import fails here, fix it before proceeding.
# Using LangChain 0.3+ import paths throughout -- legacy paths removed.
# ---------------------------------------------------------------

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os, json, time, logging, threading, requests
from typing import Optional

print("All imports validated. Environment is ready.")


# In[5]:


# ---------------------------------------------------------------
# Cell 2.1 -- Define mock tools
#
# In production, get_order_status would query your orders database
# and get_product_info would call your product catalog API.
# The function signature and docstring stay identical -- only the
# implementation changes. The agent never knows the difference.
# ---------------------------------------------------------------

from langchain_core.tools import tool

@tool
def get_order_status(order_id: str) -> str:
    """
    Get the current status of a customer order by order ID.

    Args:
        order_id: The unique order identifier (e.g., ORD00126)
    Returns:
        Human-readable string describing the current order status.
    """
    mock_orders = {
        "ORD00126": "Shipped -- Out for delivery, Delivery by 8 PM tonight.",
        "ORD00226": "Processing -- Still not shipped.PLease check in 24 hours",
        "ORD00326": "Delivered -- Delivered on May 26 at 3:30 PM.",
        "ORD00426": "Cancelled -- Refund will get creditted in 5-7 business days.",
        "ORD00526": "Return Initiated -- Please deliver at nearest Return location.",
    }
    return mock_orders.get(
        order_id.strip().upper(),
        f"Order ID '{order_id}' not found. Please verify the order ID from your confirmation email."
    )


@tool
def get_product_info(product_name: str) -> str:
    """
    Get product details including price, availability, and warranty.

    Args:
        product_name: Product name or search term (e.g., laptop, headphones)
    Returns:
        Human-readable string with product details.
    """
    mock_catalog = {
        "laptop":     "Lenono X1 -- $1456.56 -- In Stock -- 1 year on-site warranty",
        "phone":      "Samsung S24 -- $858.90 -- In Stock -- 6 months warranty",
        "headphones": "JBL-- $48.55 -- Out of Stock -- Restock expected in 1 week",
        "tablet":     "TabMax 10 -- $499 -- In Stock -- 1 year warranty",
        "charger":    "FastCharge 65W -- $19.99 -- In Stock -- 3 month warranty",
        "mouse":      "ErgoClick Wireless -- $14.99 -- In Stock -- 1 year warranty",
    }
    query = product_name.strip().lower()
    for key, value in mock_catalog.items():
        if key in query:
            return value
    return (
        f"No exact match found for '{product_name}'. "
        "Available categories: laptop, phone, headphones, tablet, charger, mouse."
    )


# Verify both tools work before wiring into the agent
print("Tool: get_order_status")
print(get_order_status.invoke({'order_id': 'ORD00126'}))
print()
print("Tool: get_product_info")
print(get_product_info.invoke({'product_name': 'laptop'}))
print()
print("Both tools validated.")


# In[6]:


# ---------------------------------------------------------------
# Cell 2.2 -- Initialise the LangGraph ReAct agent
#
# ReAct loop: Reason -> Act (tool call) -> Observe -> Reason -> ...
# The loop runs until the LLM decides it has enough information to respond.
#
# MemorySaver keys conversation history by thread_id.
# Each customer session is fully isolated.
# ---------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = (
    "You are Computer_Accessories, a professional customer support agent for BestElectronicsBuy -- "
    "an American e-commerce platform selling electronics.\n\n"
    "Your responsibilities:\n"
    "- Help customers track orders using the get_order_status tool\n"
    "- Provide product information using the get_product_info tool\n"
    "- Answer general queries about shipping, returns, and warranties\n\n"
    "Your boundaries:\n"
    "- Do not discuss topics unrelated to ShopEasy products and orders\n"
    "- If you cannot help, direct the customer to our website\n"
    "- Always respond in the same language the customer uses\n"
    "- Be concise, professional, and empathetic"
)

# gpt-4o-mini: optimal cost-to-performance ratio for customer support
# temperature=0: deterministic responses, essential for support consistency
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# MemorySaver: in-process, per-thread conversation history
# Production alternative: PostgresSaver or RedisSaver for multi-instance deployments
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[get_order_status, get_product_info],
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

print("compute_access agent initialised.")
print(f"Model: gpt-4o-mini | Tools: {['get_order_status', 'get_product_info']}")
print("Memory: MemorySaver (isolated per session_id)")


# In[7]:


import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Check what key the agent is actually seeing at runtime
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

print(f"Key being used by LLM : {os.getenv('OPENAI_API_KEY')[:12]}...")

# Make a minimal test call
try:
    response = llm.invoke("Say the word OK and nothing else.")
    print(f"LLM response : {response.content}")
    print("✅ LLM is working correctly")
except Exception as e:
    print(f"❌ LLM error : {e}")


# In[8]:


# ---------------------------------------------------------------
# Cell 2.3 -- Helper function to invoke the agent
#
# This is the single call interface between the API layer and the agent.
# Keeping this as a standalone function means it can be called from:
#   - FastAPI endpoints
#   - Test cells in this notebook
#   - Future evaluation scripts (Week 17)
# All through the same code path.
# ---------------------------------------------------------------

import time
from langchain_core.messages import HumanMessage

def invoke_compute_access(session_id: str, user_message: str) -> dict:
    """
    Invoke the compute_access agent for a given session and message.

    Args:
        session_id: Unique customer session identifier.
                    LangGraph uses this to retrieve and store conversation history.
        user_message: The customer query text.
    Returns:
        dict with keys: session_id (str), response (str), latency_ms (float)
    """
    start_time = time.time()
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )

    # The final assistant response is always the last message in the list
    response_text = result["messages"][-1].content
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "session_id": session_id,
        "response": response_text,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------
# Direct notebook test -- confirms the agent works before any
# API or deployment layer is involved
# ---------------------------------------------------------------

print("Testing compute_access directly (no API layer)...")
print("=" * 65)

test_cases = [
    ("session_demo_1", "Hi, what is the status of my order ORD00126?"),
    ("session_demo_1", "What about ORD00326?"),        # Same session -- tests memory
    ("session_demo_2", "Tell me about your laptops"),  # New session -- isolated
]

for session_id, message in test_cases:
    result = invoke_compute_access(session_id, message)
    print(f"Session : {result['session_id']}")
    print(f"Query   : {message}")
    print(f"Response: {result['response']}")
    print(f"Latency : {result['latency_ms']} ms")
    print("-" * 65)

