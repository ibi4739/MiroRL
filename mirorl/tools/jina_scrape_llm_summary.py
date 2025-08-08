# Copyright 2025 MiroMind Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import time
from typing import Any, Dict

import requests
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Jina Scraper")


@mcp.tool()
def scrape_and_extract_info(
    url: str, info_to_extract: str, custom_headers: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Scrape content from a URL and extract specific types of information using LLM.

    Args:
        url (str): The URL to scrape content from
        info_to_extract (str): The specific types of information to extract (usually a question)
        custom_headers (Dict[str, str]): Additional headers to include in the scraping request

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - url (str): The original URL
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - scrape_stats (Dict): Statistics about the scraped content
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """
    if _is_huggingface_dataset_or_space_url(url):
        return {
            "success": False,
            "url": url,
            "extracted_info": "",
            "error": "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose.",
            "scrape_stats": {},
            "tokens_used": 0,
        }

    # First, scrape the content
    scrape_result = scrape_url_with_jina(url, custom_headers)

    if not scrape_result["success"]:
        logger.error(
            f"Jina Scrape and Extract Info: Scraping failed: {scrape_result['error']}"
        )
        return {
            "success": False,
            "url": url,
            "extracted_info": "",
            "error": f"Scraping failed: {scrape_result['error']}",
            "scrape_stats": {},
            "tokens_used": 0,
        }

    # Then, summarize the content
    extracted_result = extract_info_with_llm(
        url=url,
        content=scrape_result["content"],
        info_to_extract=info_to_extract,
        max_tokens=8192,
    )

    # Combine results
    return {
        "success": extracted_result["success"],
        "url": url,
        "extracted_info": extracted_result["extracted_info"],
        "error": extracted_result["error"],
        "scrape_stats": {
            "line_count": scrape_result["line_count"],
            "char_count": scrape_result["char_count"],
            "last_char_line": scrape_result["last_char_line"],
            "all_content_displayed": scrape_result["all_content_displayed"],
        },
        "model_used": extracted_result["model_used"],
        "tokens_used": extracted_result["tokens_used"],
    }


def _is_huggingface_dataset_or_space_url(url):
    """
    Check if the URL is a Hugging Face dataset or space URL.
    :param url: The URL to check
    :return: True if it's a HuggingFace dataset or space URL, False otherwise
    """
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


def scrape_url_with_jina(
    url: str, custom_headers: Dict[str, str] = None, max_chars: int = 102400 * 4
) -> Dict[str, Any]:
    """
    Scrape content from a URL and save to a temporary file. Need to read the content from the temporary file.


    Args:
        url (str): The URL to scrape content from
        custom_headers (Dict[str, str]): Additional headers to include in the request
        max_chars (int): Maximum number of characters to reserve for the scraped content

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - filename (str): Absolute path to the temporary file containing the scraped content
            - content (str): The scraped content of the first 40k characters
            - error (str): Error message if the operation failed
            - line_count (int): Number of lines in the scraped content
            - char_count (int): Number of characters in the scraped content
            - last_char_line (int): Line number where the last displayed character is located
            - all_content_displayed (bool): Signal indicating if all content was displayed (True if content <= 40k chars)


    """

    # Validate input
    if not url or not url.strip():
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "URL cannot be empty",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get API key from environment
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "JINA_API_KEY environment variable is not set",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Construct the Jina.ai API URL
    jina_url = f"https://r.jina.ai/{url}"

    try:
        # Prepare headers
        headers = {"Authorization": f"Bearer {jina_api_key}"}

        # Add custom headers if provided
        if custom_headers:
            headers.update(custom_headers)

        # Retry configuration
        retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(retry_delays, 1):
            try:
                # Make the request using requests library
                response = requests.get(
                    jina_url,
                    headers=headers,
                    timeout=(20, 60),  # (connect timeout, read timeout)
                    allow_redirects=True,  # Follow redirects (equivalent to curl -L)
                )

                # Check if request was successful
                response.raise_for_status()
                break  # Success, exit retry loop

            except requests.exceptions.ConnectTimeout as e:
                # connection timeout, retry
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    logger.warning(
                        f"Jina Scrape: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except requests.exceptions.ConnectionError as e:
                # connection error, retry
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    logger.warning(
                        f"Jina Scrape: Connection error: {e}, {delay}s before next attempt"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Connection retry attempts exhausted, url: {url}"
                    )
                    raise e

            except requests.exceptions.ReadTimeout as e:
                # read timeout, retry
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    logger.warning(
                        f"Jina Scrape: Read timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Read timeout retry attempts exhausted, url: {url}"
                    )
                    raise e

            except requests.exceptions.HTTPError as e:
                if attempt < len(retry_delays):
                    logger.warning(
                        f"Jina Scrape: HTTP error: {e}, response.text: {response.text}, url: {url}, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: HTTP error retry attempts exhausted, url: {url}"
                    )
                    raise e

            except requests.exceptions.RequestException as e:
                if attempt < len(retry_delays):
                    logger.warning(
                        f"Jina Scrape: Unknown request exception: {e}, url: {url}, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Jina Scrape: Unknown request exception retry attempts exhausted, url: {url}"
                    )
                    raise e

    except Exception as e:
        error_msg = f"Jina Scrape: Unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": error_msg,
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get the scraped content
    content = response.text

    if not content:
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "No content returned from Jina.ai API",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # handle insufficient balance error
    try:
        content_dict = json.loads(content)
    except json.JSONDecodeError:
        content_dict = None
    if (
        isinstance(content_dict, dict)
        and content_dict.get("name") == "InsufficientBalanceError"
    ):
        return {
            "success": False,
            "filename": "",
            "content": "",
            "error": "Insufficient balance",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    # Get content statistics
    total_char_count = len(content)
    total_line_count = content.count("\n") + 1 if content else 0

    # Extract first max_chars characters
    displayed_content = content[:max_chars]
    all_content_displayed = total_char_count <= max_chars

    # Calculate the line number of the last character displayed
    if displayed_content:
        # Count newlines up to the last displayed character
        last_char_line = displayed_content.count("\n") + 1
    else:
        last_char_line = 0

    return {
        "success": True,
        "content": displayed_content,
        "error": "",
        "line_count": total_line_count,
        "char_count": total_char_count,
        "last_char_line": last_char_line,
        "all_content_displayed": all_content_displayed,
    }


EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def get_prompt_with_truncation(
    info_to_extract: str, content: str, truncate_last_num_chars: int = -1
) -> str:
    if truncate_last_num_chars > 0:
        content = content[:-truncate_last_num_chars] + "[...truncated]"

    # Prepare the prompt
    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    return prompt


def extract_info_with_llm(
    url: str,
    content: str,
    info_to_extract: str,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    Summarize content using an LLM API.

    Args:
        content (str): The content to summarize
        info_to_extract (str): The specific types of information to extract (usually a question)
        max_tokens (int): Maximum tokens for the response

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """

    # Get summary llm name from environment
    summary_llm_name = os.getenv("SUMMARY_LLM_NAME")
    if not summary_llm_name:
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_NAME environment variable is not set",
            "model_used": summary_llm_name,
            "tokens_used": 0
        }
    
    # Get summary llm url from environment
    summary_llm_url = os.getenv("SUMMARY_LLM_URL")
    if not summary_llm_url:
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_URL environment variable is not set",
            "model_used": summary_llm_name,
            "tokens_used": 0
        }

    # Validate input
    if not content or not content.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "Content cannot be empty",
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }

    prompt = get_prompt_with_truncation(info_to_extract, content)

    # Prepare the payload
    payload = {
        "model": summary_llm_name,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        # "temperature": 0.7,
        # "top_p": 0.8,
        # "top_k": 20,
    }

    # Prepare headers
    headers = {"Content-Type": "application/json"}

    try:
        # Retry configuration
        connect_retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(connect_retry_delays, 1):
            try:
                # Make the API request using requests
                response = requests.post(
                    summary_llm_url,
                    headers=headers,
                    json=payload,
                    timeout=(30, 300),  # (connect timeout, read timeout)
                )

                # Check if the request was successful
                if (
                    "Requested token count exceeds the model's maximum context length"
                    in response.text
                    or "longer than the model's context length" in response.text
                ):
                    prompt = get_prompt_with_truncation(
                        info_to_extract,
                        content,
                        truncate_last_num_chars=40960 * attempt,
                    )  # remove 40k * num_attempts chars from the end of the content
                    payload["messages"][0]["content"] = prompt
                    continue  # no need to raise error here, just try again

                response.raise_for_status()
                break  # Success, exit retry loop

            except requests.exceptions.ConnectTimeout as e:
                # connection timeout, retry
                if attempt < len(connect_retry_delays):
                    delay = connect_retry_delays[attempt]
                    logger.warning(
                        f"Jina Scrape and Extract Info: Connection timeout, {delay}s before next attempt (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except requests.exceptions.ConnectionError as e:
                # connection error, retry
                if attempt < len(connect_retry_delays):
                    delay = connect_retry_delays[attempt]
                    logger.warning(
                        f"Jina Scrape and Extract Info: Connection error: {e}, {delay}s before next attempt"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except requests.exceptions.ReadTimeout as e:
                # read timeout, LLM API is too slow, no need to retry
                if attempt < len(connect_retry_delays):
                    logger.warning(
                        f"Jina Scrape and Extract Info: LLM API attempt {attempt} read timeout"
                    )
                    continue
                else:
                    logger.error(
                        f"Jina Scrape and Extract Info: LLM API read timeout retry attempts exhausted, please check the request complexity, information to extract: {info_to_extract}, length of content: {len(content)}, url: {url}"
                    )
                    raise e

            except requests.exceptions.HTTPError as e:
                logger.error(
                    f"Jina Scrape and Extract Info: HTTP error for LLM API: {e}, response.text: {response.text}"
                )
                raise requests.exceptions.HTTPError(
                    f"response.text: {response.text}"
                ) from e

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Jina Scrape and Extract Info: Unknown request exception: {e}"
                )
                raise e

    except Exception as e:
        error_msg = f"Jina Scrape and Extract Info: Unexpected error during LLM API call: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }

    # Parse the response
    try:
        response_data = response.json()

    except json.JSONDecodeError as e:
        error_msg = (
            f"Jina Scrape and Extract Info: Failed to parse LLM API response: {str(e)}"
        )
        logger.error(error_msg)
        logger.error(f"Raw response: {response.text}")
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }

    logger.info(
        f"Jina Scrape and Extract Info: Info to extract: {info_to_extract}, LLM Response data: {response_data}"
    )

    # Extract summary from response
    if "choices" in response_data and len(response_data["choices"]) > 0:
        try:
            summary = response_data["choices"][0]["message"]["content"]
        except Exception as e:
            error_msg = f"Jina Scrape and Extract Info: Failed to get summary from LLM API response: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "extracted_info": "",
                "error": error_msg,
                "model_used": summary_llm_name,
                "tokens_used": 0,
            }

        # Extract token usage if available
        tokens_used = 0
        if "usage" in response_data:
            tokens_used = response_data["usage"].get("total_tokens", 0)

        return {
            "success": True,
            "extracted_info": summary,
            "error": "",
            "model_used": summary_llm_name,
            "tokens_used": tokens_used,
        }
    elif "error" in response_data:
        error_msg = (
            f"Jina Scrape and Extract Info: LLM API error: {response_data['error']}"
        )
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }
    else:
        error_msg = "Jina Scrape and Extract Info: No valid response from LLM API"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }


if __name__ == "__main__":
    # Example usage and testing

    # Run the MCP server
    mcp.run(transport="stdio")
