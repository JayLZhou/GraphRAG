from hashlib import md5
import html
from typing import Any, List
import re
import numbers
from Core.Common.Logger import logger
import tiktoken
from tenacity import RetryCallState, RetryError, _utils
import numpy as np
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def split_string_by_multi_markers(
    text: str, delimiters: list[str]
) -> list[str]:
    """
    Split a string by multiple delimiters.

    Args:
        text (str): The string to split.
        delimiters (list[str]): A list of delimiter strings.

    Returns:
        list[str]: A list of strings, split by the delimiters.
    """
    if not delimiters:
        return [text]
    split_pattern = "|".join(re.escape(delimiter) for delimiter in delimiters)
    segments = re.split(split_pattern, text)
    return [segment.strip() for segment in segments if segment.strip()]


def is_float_regex(value: str) -> bool:
    """
    Check if a string matches the regular expression for a float.

    Args:
        value (str): The string to check.

    Returns:
        bool: Whether the string matches the regular expression.
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


# Json operations

import json
import os
def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)
    

def community_report_from_json(parsed_output: dict) -> str:
    """Generate a community report string from parsed JSON output.

    Args:
        parsed_output (dict): A dictionary containing keys 'title', 'summary', and 'findings'.
                              'findings' is expected to be a list of dictionaries or strings.

    Returns:
        str: A formatted string representing the community report.
    """
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    report_sections = []
    for finding in findings:
        if isinstance(finding, str):
            report_sections.append(f"## {finding}\n")
        elif isinstance(finding, dict):
            summary = finding.get("summary", "")
            explanation = finding.get("explanation", "")
            report_sections.append(f"## {summary}\n\n{explanation}")

    return f"# {title}\n\n{summary}\n\n" + "\n\n".join(report_sections)



def list_to_quoted_csv_string(data: List[List[Any]]) -> str:
        """Converts a list of lists into a CSV formatted string with quoted values."""
        def enclose_string_with_quotes(content: Any) -> str:
            if isinstance(content, numbers.Number):
                return str(content)
            content = str(content).strip().strip("'").strip('"')
            return f'"{content}"'
        
        return "\n".join(
            [
                ",\t".join([enclose_string_with_quotes(data_dd) for data_dd in data_d])
                for data_d in data
            ]
        )

def parse_value_from_string(value: str):
    """
    Parse a value from a string, attempting to convert it into the appropriate type.
    
    Args:
        value: The string value to parse.
    
    Returns:
        The value converted to its appropriate type (e.g., int, float, bool, str).
    """
    try:
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        elif value.isdigit():
            return int(value)
        else:
            return float(value) if '.' in value else value.strip('"')
    except ValueError:
        return value


def prase_json_from_response(response: str) -> dict:
    """
    Extract JSON data from a string response.
    
    This function attempts to extract the first complete JSON object from the response.
    If that fails, it tries to extract key-value pairs from a potentially malformed JSON string.
    
    Args:
        response: The string response containing JSON data.    
    Returns:
        A dictionary containing the extracted JSON data.
    """
    stack = []
    first_json_start = None

    # Attempt to extract the first complete JSON object using a stack to track braces
    for i, char in enumerate(response):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = response[first_json_start:i+1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        break
                    finally:
                        first_json_start = None

    # If extraction of complete JSON failed, try extracting key-value pairs from a non-standard JSON string
    extracted_values = {}
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, response, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = prase_json_from_response(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value_from_string(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    else:
        logger.info("JSON data successfully extracted.")

    return extracted_values



def encode_string_by_tiktoken(content: str, model_name: str = "cl100k_base"):
    
    ENCODER = tiktoken.get_encoding(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
        """Truncate a list of data based on the token size."""
        # Default: cl100k_base 
        if max_token_size <= 0:
            return []
        tokens = 0
        result = []
        for data in list_data:
            token_count = len(encode_string_by_tiktoken(key(data)))
            if tokens + token_count > max_token_size:
                break
            tokens += token_count
            result.append(data)
        return result    


def processing_phrases(phrase: str) -> str:
    """
    Process a phrase string, removing non-alphanumeric characters and converting to lowercase.
    
    Args:
        phrase (str): The string to process.
    
    Returns:
        str: The processed string.
    """
    return re.sub('[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()


def min_max_normalize(x):
    """
    Min-max normalization of a list of values.

    Args: 
        x (list): A list of values to normalize.
        Returns: A list of normalized values.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)
    


def log_and_reraise(retry_state: RetryCallState):
    logger.error(f"Retry attempts exhausted. Last exception: {retry_state.outcome.exception()}")
    logger.warning(
        """
Recommend going to https://deepwisdom.feishu.cn/wiki/MsGnwQBjiif9c3koSJNcYaoSnu4#part-XdatdVlhEojeAfxaaEZcMV3ZniQ
See FAQ 5.8
"""
    )
    raise retry_state.outcome.exception()


def any_to_str_set(val) -> set:
    """Convert any type to string set."""
    res = set()

    # Check if the value is iterable, but not a string (since strings are technically iterable)
    if isinstance(val, (dict, list, set, tuple)):
        # Special handling for dictionaries to iterate over values
        if isinstance(val, dict):
            val = val.values()

        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))

    return res