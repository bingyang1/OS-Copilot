from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='.env', override=True)

SEARCH_RESULT_LIST_CHUNK_SIZE = 3
RESULT_TARGET_PAGE_PER_TEXT_COUNT = 500


class ImageSearchAPI:
    def __init__(self) -> None:
        self._mkt = 'en-US'

    def search_image(self, key_words: str, top_k: int = 10, max_retry: int = 3):
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

if __name__ == "__main__":
    res = ImageSearchAPI().search_image("IU")
    print(res)

