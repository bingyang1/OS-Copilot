from fastapi import APIRouter, HTTPException
from pydantic import BaseModel,Field
from typing import Optional
from .duck_api import DuckDuckGoAPI
from .image_search_api import ImageSearchAPI
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', override=True)

# Calculate the number of tokens in the webpage content for GPT-4. If there are too many tokens, use GPT-3.5 for summarization or search the vector database for the most relevant segment.
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

router = APIRouter()

duck_api = DuckDuckGoAPI()
image_search_api = ImageSearchAPI() 

class QueryItemV2(BaseModel):
    query: str
    top_k: Optional[int] = Field(None)
class PageItemV2(BaseModel):
    url: str
    query: Optional[str] = Field(None)

@router.get("/tools/duck/image_search", summary="Searches for images related to the provided keywords using the DuckDuckGo Image Search API. It allows specifying the number of images to return (top_k) and retries the search up to a specified number of times (max_retry) in case of failures. The search is performed with a moderate safe search filter and is intended for use within an environments that requires image search capabilities. The function returns a list of images, including their names, imageURLs,title,source and source URL,height and width, and thumbnail information. If the search fails after the maximum number of retries, it raises a runtime error.")
async def image_search(item: QueryItemV2):
    try:
        if item.top_k == None:
            item.top_k = 10
        search_results = image_search_api.search_image(item.query,item.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return search_results

@router.get("/tools/duck/video_search", summary="Searches for videos related to the provided keywords using the DuckDuckGo Video Search API. It allows specifying the number of videos to return (top_k) and retries the search up to a specified number of times (max_retry) in case of failures. The search is performed with a moderate safe search filter and is intended for use within an environments that requires video search capabilities. The function returns a list of video, including their titles, URLs,date,source,body, and thumbnail information. If the search fails after the maximum number of retries, it raises a runtime error.")
async def video_search(item: QueryItemV2):
    try:
        if item.top_k == None:
            item.top_k = 10
        search_results = duck_api.search_videos(item.query,item.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return search_results

@router.get("/tools/duck/search", summary="Execute DuckDuckGo Search - returns top web snippets related to the query. Avoid using complex filters like 'site:'. For detailed page content, further use the web browser tool.")
async def search(item: QueryItemV2):
    try:
        if item.top_k == None:
            item.top_k = 5
        search_results = duck_api.search(item.query,item.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return search_results

@router.get("/tools/duck/news_search", summary="Execute DuckDuckGo Search - returns top web snippets related to the query. Avoid using complex filters like 'site:'. For detailed page content, further use the web browser tool.")
async def news_search(item: QueryItemV2):
    try:
        if item.top_k == None:
            item.top_k = 5
        search_results = duck_api.search_news(item.query,item.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return search_results

@router.get("/tools/duck/load_pagev2", summary="Web browser tool for detailed content retrieval and specific information extraction from a target URL.In the case of Wikipedia, the number of tokens on such pages is often too large to load the entire page, so the 'query' parameter must be given to perform a similarity query to find the most relevant pieces of content. The 'query' parameter should be assigned with your task description to find the most relevant content of the web page.It is important that your 'query' must retain enough details about the task, such as time, location, quantity, and other information, to ensure that the results obtained are accurate enough.")
async def load_page_v2(item: PageItemV2):
    result = {"page_content": ""}
    try:
        raw_page_content = duck_api.load_page(item.url)
        page_token_num = num_tokens_from_string(raw_page_content)
        if(page_token_num <= 4096):
            result = {"page_content": raw_page_content}
        else:
            if item.query == None:
                summarized_page_content = duck_api.summarize_loaded_page(raw_page_content)
                result = {"page_content": summarized_page_content}
            else:
                attended_content = duck_api.attended_loaded_page(raw_page_content,item.query)
                result = {"page_content": attended_content}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result



if __name__ == "__main__":
    res = ImageSearchAPI().search_image("IU")
    print(res)
    res = DuckDuckGoAPI().search('IU')
    print(res)

