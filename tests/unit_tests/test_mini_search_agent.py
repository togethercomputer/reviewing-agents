from unittest.mock import MagicMock, patch

import pytest

from reviewing_agents.shared.mini_search_agent import (
    SearchResult,
    SearchResults,
    SmolAgentsTavilySearchTool,
    VisitWebpageTool,
    extract_tavily_results,
    tavily_search,
)


def test_search_result_creation():
    result = SearchResult(
        title="Test Title", link="https://example.com", content="Test content", raw_content="Raw test content"
    )

    assert result.title == "Test Title"
    assert result.link == "https://example.com"
    assert result.content == "Test content"
    assert result.raw_content == "Raw test content"


def test_search_result_str():
    result = SearchResult(title="Test", link="https://test.com", content="Content", raw_content="Raw")

    str_repr = str(result)
    assert "Title: Test" in str_repr
    assert "Link: https://test.com" in str_repr
    assert "Content: Content" in str_repr
    assert "Raw Content: Raw" in str_repr

    short_str = result.short_str()
    assert "Raw Content: Raw" not in short_str


def test_search_results_creation():
    results = [
        SearchResult(title="Title 1", link="https://1.com", content="Content 1"),
        SearchResult(title="Title 2", link="https://2.com", content="Content 2"),
    ]

    search_results = SearchResults(results=results)
    assert len(search_results.results) == 2
    assert search_results.results[0].title == "Title 1"


def test_search_results_add():
    results1 = SearchResults(results=[SearchResult(title="Title 1", link="https://1.com", content="Content 1")])
    results2 = SearchResults(results=[SearchResult(title="Title 2", link="https://2.com", content="Content 2")])

    combined = results1 + results2
    assert len(combined.results) == 2


def test_extract_tavily_results():
    mock_response = {
        "results": [
            {"title": "Test Title", "url": "https://test.com", "content": "Test content", "raw_content": "Raw content"}
        ]
    }

    results = extract_tavily_results(mock_response)
    assert len(results.results) == 1
    assert results.results[0].title == "Test Title"
    assert results.results[0].link == "https://test.com"


@patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"})
@patch("reviewing_agents.shared.mini_search_agent.TavilyClient")
def test_tavily_search_with_mock(mock_tavily_client):
    mock_client = MagicMock()
    mock_tavily_client.return_value = mock_client

    mock_client.search.return_value = {
        "results": [
            {
                "title": "Mock Result",
                "url": "https://mock.com",
                "content": "Mock content",
                "raw_content": "Mock raw content",
            }
        ]
    }

    # Clear the cache first
    from reviewing_agents.shared.mini_search_agent import _tavily_search_cached

    _tavily_search_cached.cache_clear()

    results = tavily_search("test query", max_results=1)

    assert len(results.results) == 1
    assert results.results[0].title == "Mock Result"
    mock_client.search.assert_called_once()


@patch.dict("os.environ", {}, clear=True)
def test_tavily_search_no_api_key():
    # Clear the cache first
    from reviewing_agents.shared.mini_search_agent import _tavily_search_cached

    _tavily_search_cached.cache_clear()

    with pytest.raises(ValueError, match="TAVILY_API_KEY environment variable is not set"):
        tavily_search("test query")


@patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"})
def test_smolagents_tavily_search_tool_creation():
    tool = SmolAgentsTavilySearchTool()
    assert tool.name == "tavily_search"
    assert tool.max_results == 3
    assert tool.include_raw is True


def test_smolagents_tavily_search_tool_no_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="TAVILY_API_KEY environment variable is not set"):
            SmolAgentsTavilySearchTool()


def test_visit_webpage_tool_gov_website():
    tool = VisitWebpageTool()
    result = tool.forward("https://example.gov")
    assert "You should not scrape a .gov website." in result


@patch("markdownify.markdownify")
@patch("requests.get")
def test_visit_webpage_tool_success(mock_get, mock_markdownify):
    mock_response = MagicMock()
    mock_response.text = "<html><body><h1>Test Page</h1></body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    mock_markdownify.return_value = "# Test Page"

    tool = VisitWebpageTool()
    result = tool.forward("https://test.com")

    assert "Test Page" in result
    mock_get.assert_called_once_with("https://test.com", timeout=20)


@patch("requests.get")
def test_visit_webpage_tool_timeout(mock_get):
    import requests

    mock_get.side_effect = requests.exceptions.Timeout()

    tool = VisitWebpageTool()
    result = tool.forward("https://test.com")

    assert "The request timed out" in result


@patch.dict("os.environ", {"TOGETHER_API_KEY": "test_key", "TAVILY_API_KEY": "test_key"})
@patch("reviewing_agents.shared.mini_search_agent.LiteLLMModel")
@patch("reviewing_agents.shared.mini_search_agent.CodeAgent")
def test_research_agent_creation(mock_code_agent, mock_llm_model):
    from reviewing_agents.shared.mini_search_agent import ResearchAgent

    agent = ResearchAgent()

    mock_llm_model.assert_called_once()
    mock_code_agent.assert_called_once()
    assert hasattr(agent, "run_query")
