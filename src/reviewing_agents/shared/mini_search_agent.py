import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

from smolagents import CodeAgent, LiteLLMModel, Tool
from tavily import InvalidAPIKeyError, MissingAPIKeyError, TavilyClient, UsageLimitExceededError


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages. This is useful after you have used the search tool to find information."  # noqa: E501
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
            from smolagents.utils import truncate_content
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."  # noqa: E501
            ) from e
        try:
            # Check if the URL is a .gov website
            if ".gov" in url.lower():
                return "You should not scrape a .gov website."

            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


@dataclass(frozen=True, kw_only=True)
class SearchResult:
    title: str
    link: str
    content: str
    raw_content: Optional[str] = None

    def __str__(self, include_raw=True):
        result = f"Title: {self.title}\nLink: {self.link}\nContent: {self.content}"
        if include_raw and self.raw_content:
            result += f"\nRaw Content: {self.raw_content}"
        return result

    def short_str(self):
        return self.__str__(include_raw=False)


@dataclass(frozen=True, kw_only=True)
class SearchResults:
    results: list[SearchResult]

    def __str__(self, short=False):
        if short:
            result_strs = [result.short_str() for result in self.results]
        else:
            result_strs = [str(result) for result in self.results]
        return "\n\n".join(f"[{i + 1}] {result_str}" for i, result_str in enumerate(result_strs))

    def __add__(self, other):
        return SearchResults(results=self.results + other.results)

    def short_str(self):
        return self.__str__(short=True)


# Cache for tavily search results
_tavily_cache: Dict[Tuple[str, int, bool], SearchResults] = {}


def extract_tavily_results(response) -> SearchResults:
    """Extract key information from Tavily search results."""
    results = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                content=item.get("content", ""),
                raw_content=item.get("raw_content", ""),
            )
        )
    return SearchResults(results=results)


@lru_cache(maxsize=500)
def _tavily_search_cached(query: str, max_results: int, include_raw: bool) -> SearchResults:
    """Cached implementation of tavily search."""
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")

    client = TavilyClient(api_key)

    response = client.search(query=query, max_results=max_results, include_raw_content=include_raw)

    return extract_tavily_results(response)


def tavily_search(query: str, max_results=3, include_raw: bool = True) -> SearchResults:
    """
    Perform a search using the Tavily Search API with the official client.

    Parameters:
        query (str): The search query.
        search_depth (str): The depth of search - 'basic' or 'deep'.
        max_results (int): Maximum number of results to return.

    Returns:
        list: Formatted search results with title, link, and snippet.
    """
    return _tavily_search_cached(query, max_results, include_raw)


class SmolAgentsTavilySearchTool(Tool):
    name = "tavily_search"
    description = """Performs a Tavily web search based on your query (similar to a Google search) and returns the top search results."""  # noqa: E501
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results=3, include_raw=True, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.include_raw = include_raw

        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY environment variable is not set")

    def forward(self, query: str) -> str:
        try:
            results: SearchResults = tavily_search(
                query=query, max_results=self.max_results, include_raw=self.include_raw
            )

            if len(results.results) == 0:
                raise Exception("No results found! Try a less restrictive/shorter query.")

            postprocessed_results = []
            for result in results.results:
                postprocessed_results.append(f"[{result.title}]({result.link})\n{result.content}")

            return "## Search Results\n\n" + "\n\n".join(postprocessed_results)
        except (InvalidAPIKeyError, MissingAPIKeyError, UsageLimitExceededError) as e:
            # this is a very bad way of handling this, but it's the only way I can think of to get the error to the user. # noqa: E501
            # if search fails, results will be biased.
            print("Error performing search: ", e)
            sys.exit(1)


def delay_execution_5(pagent, **kwargs) -> bool:
    """
    Delays the execution for 10 seconds.
    """
    time.sleep(1)
    return True


class ResearchAgent:
    def __init__(self, model_name="gpt-4.1", api_key=None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        model_id = model_name
        self.model = LiteLLMModel(model_id=model_id, api_key=api_key)
        tools: list[Tool] = [SmolAgentsTavilySearchTool(), VisitWebpageTool()]
        self.agent = CodeAgent(tools=tools, model=self.model, step_callbacks=[delay_execution_5])

    def run_query(self, query):
        return self.agent.run(query, reset=True)


if __name__ == "__main__":
    agent = ResearchAgent()

    query = "What are the latest developments in AI?"
    response = agent.run_query(query)
    print(f"Agent Response:\n{response}")
