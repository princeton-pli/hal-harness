from typing import Optional

import requests

# from smolagents.agents import ToolCallingAgent
from smolagents import CodeAgent, HfApiModel, tool, DuckDuckGoSearchTool, PythonInterpreterTool



# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620'
model = LiteLLMModel(model_id="gpt-4o")

@tool
def search_wikipedia(query: str) -> str:
    """
    Fetches a summary of a Wikipedia page for a given query.
    Args:
        query: The search term to look up on Wikipedia.
    Returns:
        str: A summary of the Wikipedia page if successful, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        title = data["title"]
        extract = data["extract"]

        return f"Summary for {title}: {extract}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching Wikipedia data: {str(e)}"


agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        PythonInterpreterTool(),
        search_wikipedia,
    ],
    model=model,
)

# Uncomment the line below to run the agent with a specific query


