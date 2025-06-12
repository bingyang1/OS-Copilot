import copy
import numpy as np
import itertools
import json
import logging
import os
import re
import string
from typing import Any
import tqdm
import re
import tiktoken
import random
from datasets import load_dataset
from oscopilot.prompts.general_pt import prompt as general_pt
from oscopilot.utils.llms import OpenAI
import platform
from functools import wraps


def save_json(file_path, new_json_content):
    """
    将JSON内容保存到文件中。

    Args:
    file_path (str): JSON文件的路径。
    new_json_content (dict或list): 要保存的新JSON内容。

    Returns:
        None
    """

    """
    Saves JSON content to a file.

    Args:
        file_path (str): The path to the JSON file.
        new_json_content (dict or list): The new JSON content to be saved.

    Returns:
        None
    """

    # Check if the file exists
    if os.path.exists(file_path): 
        # If the file exists, read its content
        with open(file_path, 'r') as f:
            json_content = json.load(f)
        
        # Check the type of existing JSON content
        if isinstance(json_content, list):
            # If the existing content is a list, append or extend the new content
            if isinstance(new_json_content, list):
                json_content.extend(new_json_content)    
            else:
                json_content.append(new_json_content)   
        elif isinstance(json_content, dict):  
            # If the existing content is a dictionary, update it with the new content
            if isinstance(new_json_content, dict):
                json_content.update(new_json_content)                
            else:
                # If the new content is not a dictionary, return without saving
                return
        else:
            # If the existing content is neither a list nor a dictionary, return without saving
            return
        
        # Write the updated JSON content back to the file
        with open(file_path, 'w') as f:
            json.dump(json_content, f, indent=4)
    else:
        # If the file does not exist, create a new file and write the new content to it
        with open(file_path, 'w') as f:
            json.dump(new_json_content, f, indent=4)


def read_json(file_path):
    """
    Reads JSON content from a file.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict or list: The JSON content read from the file. If the file contains a JSON object, it returns a dictionary. 
                      If the file contains a JSON array, it returns a list.
    """    
    with open(file_path, 'r') as f:
        json_content = json.load(f)
    return json_content


def random_string(length):
    """
    Generates a random string of a specified length.

    Args:
        length (int): The desired length of the random string.

    Returns:
        str: A string of random characters and digits of the specified length.
    """
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def num_tokens_from_string(string: str) -> int:
    """
    Calculates the number of tokens in a given text string according to a specific encoding.

    Args:
        text (str): The text string to be tokenized.

    Returns:
        int: The number of tokens the string is encoded into according to the model's tokenizer.
    """
    encoding = tiktoken.encoding_for_model('gpt-4-1106-preview')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def parse_content(content, html_type="html.parser"):
    """
    Parses and cleans the given HTML content, removing specified tags, ids, and classes.
    解析并清理给定的 HTML 内容，移除指定的标签、ID 和类。

    Args:
        content (str): The HTML content to be parsed and cleaned.
        type (str, optional): The type of parser to be used by BeautifulSoup. Defaults to "html.parser".
            Supported types include "html.parser", "lxml", "lxml-xml", "xml", and "html5lib".

    Raises:
        ValueError: If an unsupported parser type is specified.

    Returns:
        str: The cleaned text extracted from the HTML content.
    """
    implemented = ["html.parser", "lxml", "lxml-xml", "xml", "html5lib"]
    if html_type not in implemented:
        raise ValueError(f"Parser type {html_type} not implemented. Please choose one of {implemented}")

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, html_type)
    original_size = len(str(soup.get_text()))

    tags_to_exclude = [
        "nav",
        "aside",
        "form",
        "header",
        "noscript",
        "svg",
        "canvas",
        "footer",
        "script",
        "style",
    ]
    for tag in soup(tags_to_exclude):
        tag.decompose()

    ids_to_exclude = ["sidebar", "main-navigation", "menu-main-menu"]
    for id in ids_to_exclude:
        tags = soup.find_all(id=id)
        for tag in tags:
            tag.decompose()

    classes_to_exclude = [
        "elementor-location-header",
        "navbar-header",
        "nav",
        "header-sidebar-wrapper",
        "blog-sidebar-wrapper",
        "related-posts",
    ]
    for class_name in classes_to_exclude:
        tags = soup.find_all(class_=class_name)
        for tag in tags:
            tag.decompose()

    content = soup.get_text()
    content = clean_string(content)

    cleaned_size = len(content)
    if original_size != 0:
        logging.info(
            f"Cleaned page size: {cleaned_size} characters, down from {original_size} (shrunk: {original_size-cleaned_size} chars, {round((1-(cleaned_size/original_size)) * 100, 2)}%)"  # noqa:E501
        )

    return content


def clean_string(text):
    """
    Cleans a given string by performing various operations such as whitespace normalization,
    removal of backslashes, and replacement of hash characters with spaces. It also reduces
    consecutive non-alphanumeric characters to a single occurrence.

    通过执行各种操作（如空白字符规范化、删除反斜杠以及将井号字符替换为空格）来清理给定的字符串。它还会将连续的非字母数字字符缩减为单个字符。

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text after applying all the specified cleaning operations.
    """
    # Replacement of newline characters:
    text = text.replace("\n", " ")

    # Stripping and reducing multiple spaces to single:
    cleaned_text = re.sub(r"\s+", " ", text.strip())

    # Removing backslashes:
    cleaned_text = cleaned_text.replace("\\", "")

    # Replacing hash characters:
    cleaned_text = cleaned_text.replace("#", " ")

    # Eliminating consecutive non-alphanumeric characters:
    # This regex identifies consecutive non-alphanumeric characters (i.e., not
    # a word character [a-zA-Z0-9_] and not a whitespace) in the string
    # and replaces each group of such characters with a single occurrence of
    # that character.
    # For example, "!!! hello !!!" would become "! hello !".
    cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)

    return cleaned_text


def is_readable(s):
    """
    Heuristic to determine if a string is "readable" (mostly contains printable characters and forms meaningful words)
    用于判断一个字符串是否“可读”的启发式方法（该字符串大多包含可打印字符且能组成有意义的单词）

    :param s: string
    :return: True if the string is more than 95% printable.
    """
    try:
        printable_ratio = sum(c in string.printable for c in s) / len(s)
    except ZeroDivisionError:
        logging.warning("Empty string processed as unreadable")
        printable_ratio = 0
    return printable_ratio > 0.95  # 95% of characters are printable


def format_source(source: str, limit: int = 20) -> str:
    """
    Format a string to only take the first x and last x letters.
    This makes it easier to display a URL, keeping familiarity while ensuring a consistent length.
    If the string is too short, it is not sliced.
    格式化字符串，只取前x个和后x个字母。
    这样可以更轻松地显示URL，既保持熟悉度，又确保长度一致。
    如果字符串太短，则不进行切片。
    """
    if len(source) > 2 * limit:
        return source[:limit] + "..." + source[-limit:]
    return source


def is_valid_json_string(source: str):
    """
    Checks if a given string is a valid JSON.
    检查给定的字符串是否是有效的JSON。
    
    Args:
        source (str): The string to be validated as JSON.

    Returns:
        bool: True if the given string is a valid JSON format, False otherwise.
    """
    try:
        _ = json.loads(source)
        return True
    except json.JSONDecodeError:
        logging.error(
            "Insert valid string format of JSON. \
            Check the docs to see the supported formats - `https://docs.embedchain.ai/data-sources/json`"
        )
        return False


def chunks(iterable, batch_size=100, desc="Processing chunks"):
    """
    Breaks an iterable into smaller chunks of a specified size, yielding each chunk in sequence.
    将一个可迭代对象分割成指定大小的较小块，并按顺序生成每个块。

    Args:
        iterable (iterable): The iterable to be chunked.
        batch_size (int, optional): The size of each chunk. Defaults to 100.
        desc (str, optional): Description text to be displayed alongside the progress bar. Defaults to "Processing chunks".

    Yields:
        tuple: A chunk of the iterable, with a maximum length of `batch_size`.
    """
    it = iter(iterable)
    total_size = len(iterable)

    with tqdm(total=total_size, desc=desc, unit="batch") as pbar:
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            pbar.update(len(chunk))
            chunk = tuple(itertools.islice(it, batch_size))


def generate_prompt(template: str, replace_dict: dict):
    """
    Generates a string by replacing placeholders in a template with values from a dictionary.
    通过使用字典中的值替换模板中的占位符来生成一个字符串。

    Args:
        template (str): The template string containing placeholders to be replaced.
        replace_dict (dict): A dictionary where each key corresponds to a placeholder in the template
                             and each value is the replacement for that placeholder.

    Returns:
        str: The resulting string after all placeholders have been replaced with their corresponding values.
    """
    prompt = copy.deepcopy(template)
    for k, v in replace_dict.items():
        prompt = prompt.replace(k, str(v))
    return prompt


def cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors.
    计算两个向量之间的余弦相似度。

    Args:
        a (array_like): The first vector.
        b (array_like): The second vector.

    Returns:
        float: The cosine similarity between vectors `a` and `b`.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\
    

def send_chat_prompts(sys_prompt, user_prompt, llm, prefix=""):
    """
    Sends a sequence of chat prompts to a language learning model (LLM) and returns the model's response.
    向语言学习模型（LLM）发送一系列聊天提示，并返回模型的回复。

    Args:
        sys_prompt (str): The system prompt that sets the context or provides instructions for the language learning model.
        user_prompt (str): The user prompt that contains the specific query or command intended for the language learning model.
        llm (object): The language learning model to which the prompts are sent. This model is expected to have a `chat` method that accepts structured prompts.

    Returns:
        The response from the language learning model, which is typically a string containing the model's answer or generated content based on the provided prompts.

    The function is a utility for simplifying the process of sending structured chat prompts to a language learning model and parsing its response, useful in scenarios where dynamic interaction with the model is required.
    """
    message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return llm.chat(message, prefix=prefix)


def get_project_root_path():
    """
    This function returns the absolute path of the project root directory. It assumes that it is being called from a file located in oscopilot/utils/.
    此函数返回项目根目录的绝对路径。它假定该函数是从位于oscopilot/utils/目录下的文件中调用的。
    
    Args:
        None
    
    Returns:
        str: The absolute path of the project root directory.
    """
    script_path = os.path.abspath(__file__)

    # Get the directory of the script (oscopilot/utils)
    script_directory = os.path.dirname(script_path)

    # Get the parent directory of script_directory (oscopilot)
    oscopilot_directory = os.path.dirname(script_directory)

    # Get the project root directory
    project_root_path = os.path.dirname(oscopilot_directory)

    return project_root_path + '/'


def GAIA_postprocess(question, response):
    '''
    Postprocess the response from GAIA.
    对来自 GAIA 的回复进行后处理。
    Args:
        question (str): The question asked to GAIA.
        response (str): The response from GAIA.
    
    Returns:
        str: The postprocessed response from GAIA.

    '''
    llm = OpenAI()
    extractor_prompt = general_pt['GAIA_ANSWER_EXTRACTOR_PROMPT'].format(
        question=question,
        response=response
    )
    result = send_chat_prompts('', extractor_prompt, llm)
    return result


class GAIALoader:
    def __init__(self, level=1, cache_dir=None):
        if cache_dir != None:
            assert os.path.exists(cache_dir), f"Cache directory {cache_dir} does not exist."
            self.cache_dir = cache_dir
            try:
                self.dataset = load_dataset("gaia-benchmark/GAIA", "2023_level{}".format(level), cache_dir=self.cache_dir)
            except Exception as e:
                raise Exception(f"Failed to load GAIA dataset: {e}")
        else:
            self.dataset = load_dataset("gaia-benchmark/GAIA", "2023_level{}".format(level))
            
        
    def get_data_by_task_id(self, task_id, dataset_type):
        if self.dataset is None or dataset_type not in self.dataset:
            raise ValueError("Dataset not loaded or data set not available.")

        data_set = self.dataset[dataset_type]
        for record in data_set:
            if record['task_id'] == task_id:
                return record
        return None

    def task2query(self, task):
        query = 'Your task is: {}'.format(task['Question'])
        if task['file_name'] != '':
            query = query + '\n{0} is the absolute file path you need to use, and the file type is {1}. Note that there is no file extension at the end.'.format(task['file_path'], task['file_name'].split('.')[-1])
        print('GAIA Task {0}:\n{1}'.format(task['task_id'], query))
        logging.info(query)
        return query
    
class SheetTaskLoader:
    def __init__(self, sheet_task_path=None):
        if sheet_task_path != None:
            assert os.path.exists(sheet_task_path), f"Sheet task jsonl file {sheet_task_path} does not exist."
            self.sheet_task_path = sheet_task_path
            try:
                self.dataset = self.load_sheet_task_dataset()
            except Exception as e:
                raise Exception(f"Failed to load sheet task dataset: {e}")
        else:
            print("Sheet task jsonl file not provided.")


    def load_sheet_task_dataset(self):
        dataset = []
        with open(self.sheet_task_path, 'r') as file:
            for _, line in enumerate(file):
                task_info = json.loads(line)
                query = self.task2query(task_info['Context'], task_info['Instructions'], get_project_root_path() + task_info['file_path'])
                dataset.append(query)
        return dataset

    def task2query(self, context, instructions, file_path):
        SHEET_TASK_PROMPT = """You are an expert in handling excel file. {context}
                               Your task is: {instructions}
                               The file path of the excel is: {file_path}. Every subtask's description must include the file path, and all subtasks are completed on the file at that path.
                            """
        query = SHEET_TASK_PROMPT.format(context=context, instructions=instructions, file_path=file_path)
        return query
    
    def get_data_by_task_id(self, task_id):
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        return self.dataset[task_id]


def get_os_version():
    """
    Determines the operating system version of the current system.

    This function checks the operating system of the current environments and attempts
    to return a human-readable version string. For macOS, it uses the `platform.mac_ver()`
    method. For Linux, it attempts to read the version information from `/etc/os-release`.
    If the system is not macOS or Linux, or if the Linux version cannot be determined, it
    defaults to a generic version string or "Unknown Operating System".

    确定当前系统的操作系统版本。

    此函数检查当前环境的操作系统，并尝试返回一个易于理解的版本字符串。对于 macOS，它使用platform.mac_ver()方法。对于 Linux，它尝试从/etc/os-release读取版本信息。如果系统既不是 macOS 也不是 Linux，或者无法确定 Linux 版本，则默认为通用版本字符串或 “未知操作系统”。

    Returns:
        str: A string describing the operating system version, or "Unknown Operating System"
             if the version cannot be determined.
    """
    system = platform.system()

    if system == "Darwin":
        # macOS
        return 'macOS ' + platform.mac_ver()[0]
    elif system == "Linux":
        try:
            with open("/etc/os-release") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("PRETTY_NAME"):
                        return line.split("=")[1].strip().strip('"')
        except FileNotFoundError:
            pass

        return platform.version()
    elif system == "Windows":
        return 'Windows ' + platform.win32_ver()[0]
    else:
        return "Unknown Operating System"


def check_os_version(s):
    """
    Checks if the operating system version string matches known supported versions.

    This function examines a given operating system version string to determine if it
    contains known substrings that indicate support (e.g., "mac", "Ubuntu", "CentOS").
    If the version string does not match any of the known supported versions, it raises
    a ValueError.

    检查操作系统版本字符串是否与已知的受支持版本匹配。

    此函数检查给定的操作系统版本字符串，以确定它是否包含表明受支持的已知子字符串（例如，“mac”、“Ubuntu”、“CentOS”）。如果版本字符串与任何已知的受支持版本都不匹配，则会引发`ValueError`。 
    

    Args:
        s (str): The operating system version string to check.

    Raises:
        ValueError: If the operating system version is not recognized as a known
                    supported version.
    """
    # os = ['mac','Ubuntu','CentOS','AlmaLinux','Anolis','Windows']
    # if True in  [v in 'Windows' for v in os]:
    #     print("Operating System Version:", s)
    # else:
    #     raise ValueError("Unknown Operating System")
    
    if "mac" in s or "Ubuntu" in s or "CentOS" in s or "AlmaLinux" in s or "Anolis" in s:
        print("Operating System Version:", s)
    else:
        raise ValueError("Unknown Operating System")


def api_exception_mechanism(max_retries=3):
    """
    A decorator to add a retry mechanism to functions, particularly for handling API calls.
    This decorator will retry a function up to `max_retries` times if an exception is raised.

    一个用于为函数添加重试机制的装饰器，特别是用于处理API调用。
    这个装饰器会在函数引发异常时，最多重试 `max_retries` 次。

    Args:
    max_retries (int): The maximum number of retries allowed before giving up and re-raising the exception.

    Returns:
    function: A wrapper function that incorporates the retry mechanism.
    """
    def decorator(func):
        """
        The actual decorator that takes a function and applies the retry logic to it.

        实际的装饰器，它接受一个函数并将重试逻辑应用于该函数。

        Args:
        func (function): The function to which the retry mechanism will be applied.

        Returns:
        function: The wrapped function with retry logic.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            A wrapper function that executes the decorated function and handles exceptions by retrying.

            一个包装函数，它执行被装饰的函数，并通过重试来处理异常。

            Args:
            *args: Variable length argument list for the decorated function.
            **kwargs: Arbitrary keyword arguments for the decorated function.

            Returns:
            Any: The return value of the decorated function if successful.

            Raises:
            Exception: Re-raises any exception if the max retry limit is reached.
            """
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logging.error(f"Error on attempt {attempts} in {func.__name__}: {str(e)}")
                    if attempts == max_retries:
                        logging.error(f"Max retries reached in {func.__name__}, operation failed.")
                        raise
        return wrapper
    return decorator
