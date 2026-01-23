from smolagents import LiteLLMModel, CodeAgent, Tool, DuckDuckGoSearchTool, PythonInterpreterTool, FinalAnswerTool
import time

import smolagents.models
import re

def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.
    
    Not supported with reasoning models openai/o3, openai/o4-mini, and gpt-5 (and their versioned variants).
    """
    model_name = model_id.split("/")[-1]
    # o3, o4-mini, and gpt-5 (including versioned variants) don't support stop parameter
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*|gpt-5[-\d]*)$"
    return not re.match(pattern, model_name)

# Replace the function in smolagents
smolagents.models.supports_stop_parameter = supports_stop_parameter

AUTHORIZED_IMPORTS = [
    "os",
    "time",
    "pickle",
    "itertools",
    "random",
    "copy",
    "math",
    "cmath",
    "collections",
    "functools",
    "numpy.*",
    "scipy.*",
    "mpl_toolkits.mplot3d",
    "sympy",
    "builtins.dir",
    "builtins.slice"
]

class ModifiedWikipediaSearchTool(Tool):
    """
    Modifies WikipediaSearchTool to search for any pages relating to request, selects first page, and returns summary.

    Attributes:
        user_agent (str): A custom user-agent string to identify the project. This is required as per Wikipedia API policies, read more here: http://github.com/martin-majlis/Wikipedia-API/blob/master/README.rst
        language (str): The language in which to retrieve Wikipedia articles.
                http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (str): Defines the content to fetch. Can be "summary" for a short summary or "text" for the full article.
        extract_format (str): Defines the output format. Can be `"WIKI"` or `"HTML"`.

    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent 
        self.language = language
        self.content_type = content_type

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        try:
            import wikipedia
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia` to run this tool: for instance run `pip install wikipedia`"
            ) from e
        try:
            page = self.wiki.page(query)

            if not page.exists():
                # Try searching for related pages
                search_results = wikipedia.search(query)
                if search_results:
                    # Use the top search result
                    top_result = search_results[0]
                    page = self.wiki.page(top_result)
                    if not page.exists():
                        return f"No Wikipedia page found for '{query}', even after searching."
                else:
                    return f"No Wikipedia page found for '{query}', and no related results were found."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class RateLimitAwareDuckDuckGoSearchTool(Tool):
    """
    DuckDuckGo search tool with rate limiting awareness and fallback to Wikipedia
    """
    name = "web_search"
    description = "Searches the web using DuckDuckGo with rate limiting protection and Wikipedia fallback."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query.",
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.last_search_time = 0
        self.min_interval = 2.0  # Minimum 2 seconds between searches
        self.wikipedia_tool = ModifiedWikipediaSearchTool()
        
    def forward(self, query: str) -> str:
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        try:
            # Try DuckDuckGo first
            ddg_tool = DuckDuckGoSearchTool()
            result = ddg_tool.forward(query)
            self.last_search_time = time.time()
            
            # Check if we got a rate limit error
            if "202 Ratelimit" in result or "rate limit" in result.lower():
                # Fallback to Wikipedia
                return f"⚠️ Web search rate limited. Falling back to Wikipedia.\n\n{self.wikipedia_tool.forward(query)}"
            
            return result
            
        except Exception as e:
            # If DuckDuckGo fails, fallback to Wikipedia
            error_msg = str(e)
            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                return f"⚠️ Web search rate limited. Falling back to Wikipedia.\n\n{self.wikipedia_tool.forward(query)}"
            else:
                return f"⚠️ Web search failed ({error_msg}). Falling back to Wikipedia.\n\n{self.wikipedia_tool.forward(query)}"
        
        
def get_agent(model_params) -> CodeAgent:
    """
    Returns a CodeAgent with the specified model name.
    
    Args:
        model_name (str): The name of the model to use.
        
    Returns:
        CodeAgent: An instance of CodeAgent configured with the specified model.
    """
    # Initialize the LiteLLMModel with the specified model name
    model = LiteLLMModel(**model_params)

    # Create a CodeAgent instance with the specified model
    agent = CodeAgent(
        tools=[
            RateLimitAwareDuckDuckGoSearchTool(),
            PythonInterpreterTool(),
            ModifiedWikipediaSearchTool(),
            FinalAnswerTool(description = "Produce the final answer to the problem as a code chunk with function described in the problem description. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps. Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification. DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```. THIS IS EXTREMELY IMPORTANT! DO NOT SUBMIT A RESPONSE THAT IS NOT A VALID PYTHON CODE BLOCK!\n\nIMPORTANT COMPATIBILITY NOTES:\n- scipy.integrate.simps has been deprecated. Use scipy.integrate.simpson instead.\n- Always use numpy arrays for numerical computations.\n- If you need to use dir() or slice(), they are available as builtin functions.\n- Handle matrix operations carefully and implement missing functionality when needed.")
        ],
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        model=model,
        planning_interval=3,
        max_steps=5,
        verbosity_level=2
    )

    return agent

