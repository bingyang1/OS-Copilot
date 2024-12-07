import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from typing import Tuple
from enum import Enum
from .web_loader import WebPageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env', override=True)


SEARCH_RESULT_LIST_CHUNK_SIZE = 3
RESULT_TARGET_PAGE_PER_TEXT_COUNT = 500


class DuckDuckGoAPI:
    """
    用于与duckduckgo搜索API交互并对网页数据执行后续处理的类。

    该类封装了使用duckduckgo API执行网络搜索、加载网页，对文本进行组块和嵌入分析，总结网页，并根据特定查询处理加载的页面。

    属性：
        search_engine（BingSearchAPIWrapper）：已配置用于使用Bing的API执行搜索的实例。
        web_loader（WebPageLoader）：用于加载网页内容的实用程序。
        web_chunker（RecursiveCharacterTextSplitter）：用于将文本拆分为可管理块的实用程序。
        web_snitter_embed（OpenAIEmbeddings）：文本块的嵌入模型。
        web_summarizer（OpenAI）：用于总结网页内容的模型。

    """
    def __init__(self) -> None:
        """
        Initializes the BingAPIV2 with components for search, web page loading, and text processing.
        """
        self.web_loader = WebPageLoader()
        self.web_chunker = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=0)
        self.web_sniptter_embed = OpenAIEmbeddings()
        self.web_summarizer = OpenAI(
            temperature=0,
            )

    def search_news(self, key_words: str, top_k: int = 10, max_retry: int = 3):
        """
        Searches for images using DuckDuckGo's API with retry and proxy support.

        Args:
            key_words (str): The search query.
            top_k (int): The maximum number of results to fetch. Default is 10.
            max_retry (int): The number of retries for each mode (direct or proxy). Default is 3.

        Returns:
            list: A list of image search results.

        Raises:
            RuntimeError: If the API fails to return results after all retries.
        """
        result = []
        proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY"),
        }

        # Helper function to perform search
        def perform_search(proxy_enabled=False):
            with DDGS(proxies=proxies if proxy_enabled else None) as ddgs:
                return ddgs.news(key_words, region=self._mkt, max_results=top_k)

        # Try without proxy first
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=False)
                if result:
                    return result
            except Exception as e:
                print(f"Direct search attempt failed: {e}")
                continue

        # Fallback to using proxy
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=True)
                if result:
                    return result
            except Exception as e:
                print(f"Proxy search attempt failed: {e}")
                continue

        raise RuntimeError("Failed to access DuckDuckGo Search API after all retries.")

    def search_videos(self, key_words: str, top_k: int = 10, max_retry: int = 3):
        """
        Searches for images using DuckDuckGo's API with retry and proxy support.

        Args:
            key_words (str): The search query.
            top_k (int): The maximum number of results to fetch. Default is 10.
            max_retry (int): The number of retries for each mode (direct or proxy). Default is 3.

        Returns:
            list: A list of image search results.

        Raises:
            RuntimeError: If the API fails to return results after all retries.
        """
        result = []
        proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY"),
        }

        # Helper function to perform search
        def perform_search(proxy_enabled=False):
            with DDGS(proxies=proxies if proxy_enabled else None) as ddgs:
                return ddgs.videos(key_words, region=self._mkt, max_results=top_k)

        # Try without proxy first
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=False)
                if result:
                    return result
            except Exception as e:
                print(f"Direct search attempt failed: {e}")
                continue

        # Fallback to using proxy
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=True)
                if result:
                    return result
            except Exception as e:
                print(f"Proxy search attempt failed: {e}")
                continue

        raise RuntimeError("Failed to access DuckDuckGo Search API after all retries.")

    def search_images(self, key_words: str, top_k: int = 10, max_retry: int = 3):
        """
        Searches for images using DuckDuckGo's API with retry and proxy support.

        Args:
            key_words (str): The search query.
            top_k (int): The maximum number of results to fetch. Default is 10.
            max_retry (int): The number of retries for each mode (direct or proxy). Default is 3.

        Returns:
            list: A list of image search results.

        Raises:
            RuntimeError: If the API fails to return results after all retries.
        """
        result = []
        proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY"),
        }

        # Helper function to perform search
        def perform_search(proxy_enabled=False):
            with DDGS(proxies=proxies if proxy_enabled else None) as ddgs:
                return ddgs.images(key_words, region=self._mkt, max_results=top_k)

        # Try without proxy first
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=False)
                if result:
                    return result
            except Exception as e:
                print(f"Direct search attempt failed: {e}")
                continue

        # Fallback to using proxy
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=True)
                if result:
                    return result
            except Exception as e:
                print(f"Proxy search attempt failed: {e}")
                continue

        raise RuntimeError("Failed to access DuckDuckGo Search API after all retries.")

    def search(self, key_words: str, top_k: int = 10, max_retry: int = 3):
        """
        Searches for images using DuckDuckGo's API with retry and proxy support.

        Args:
            key_words (str): The search query.
            top_k (int): The maximum number of results to fetch. Default is 10.
            max_retry (int): The number of retries for each mode (direct or proxy). Default is 3.

        Returns:
            list: A list of image search results.

        Raises:
            RuntimeError: If the API fails to return results after all retries.
        """
        result = []
        proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY"),
        }

        # Helper function to perform search
        def perform_search(proxy_enabled=False):
            with DDGS(proxies=proxies if proxy_enabled else None) as ddgs:
                return ddgs.text(key_words, region=self._mkt, max_results=top_k)

        # Try without proxy first
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=False)
                if result:
                    return result
            except Exception as e:
                print(f"Direct search attempt failed: {e}")
                continue

        # Fallback to using proxy
        for _ in range(max_retry):
            try:
                result = perform_search(proxy_enabled=True)
                if result:
                    return result
            except Exception as e:
                print(f"Proxy search attempt failed: {e}")
                continue

        raise RuntimeError("Failed to access DuckDuckGo Search API after all retries.")

    def load_page(self, url: str) -> str:
        """
        Loads the content of a web page given its URL.

        Args:
            url (str): The URL of the web page to load.

        Returns:
            str: The content of the web page as a string.
        """
        page_data = self.web_loader.load_data(url)
        page_content_str = ""
        if(page_data["data"][0] != None and page_data["data"][0]["content"] != None):
            page_content_str = page_data["data"][0]["content"]
        return page_content_str
    def summarize_loaded_page(self,page_str):
        """
        Summarizes the content of a loaded web page.

        Args:
            page_str (str): The content of the web page to summarize.

        Returns:
            str: The summarized content of the web page.
        """
        if page_str == "":
            return ""
        web_chunks = self.web_chunker.create_documents([page_str])
        summarize_chain = load_summarize_chain(self.web_summarizer, chain_type="map_reduce")
        main_web_content = summarize_chain.run(web_chunks)
        return main_web_content
    def attended_loaded_page(self,page_str,query_str):
        """
        Identifies and aggregates content from a loaded web page that is most relevant to a given query.

        Args:
            page_str (str): The content of the web page.
            query_str (str): The query string to identify relevant content.

        Returns:
            str: The aggregated content from the web page that is most relevant to the query.
        """
        if page_str == "":
            return ""
        web_chunks = self.web_chunker.create_documents([page_str])
        chunSearch = Chroma.from_documents(web_chunks, self.web_sniptter_embed)
        relatedChunks = chunSearch.similarity_search(query_str, k=3)
        attended_content = '...'.join([chunk.page_content for chunk in relatedChunks])
        return attended_content

if __name__ == "__main__":
    print(DuckDuckGoAPI().search("IU"))
