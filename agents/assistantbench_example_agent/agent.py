import os
from smolagents import LiteLLMModel, CodeAgent, Tool, DuckDuckGoSearchTool, VisitWebpageTool

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

model = LiteLLMModel(model_id="gpt-4o-mini",
                     api_key = os.environ['OPENAI_API_KEY'])

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        ModifiedWikipediaSearchTool(),
    ],
    model=model,
    max_steps = 20,
    verbosity_level = 2
)


