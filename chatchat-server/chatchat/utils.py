from functools import partial
import logging
import os
import time

import loguru
import loguru._logger
from memoization import cached, CachingAlgorithmFlag
from chatchat.settings import Settings


from typing import (
    Union,
    Tuple,
    Dict,
    List,
    Callable,
    Generator,
    Any,
    Awaitable
)
import aiohttp
import asyncio
import base64
import json
import re
import httpx
import urllib.parse
from html2text import HTML2Text
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _filter_logs(record: dict) -> bool:
    # hide debug logs if Settings.basic_settings.log_verbose=False 
    if record["level"].no <= 10 and not Settings.basic_settings.log_verbose:
        return False
    # hide traceback logs if Settings.basic_settings.log_verbose=False 
    if record["level"].no == 40 and not Settings.basic_settings.log_verbose:
        record["exception"] = None
    return True


# 默认每调用一次 build_logger 就会添加一次 hanlder，导致 chatchat.log 里重复输出
@cached(max_size=100, algorithm=CachingAlgorithmFlag.LRU)
def build_logger(log_file: str = "chatchat"):
    """
    build a logger with colorized output and a log file, for example:

    logger = build_logger("api")
    logger.info("<green>some message</green>")

    user can set basic_settings.log_verbose=True to output debug logs
    use logger.exception to log errors with exceptions
    """
    loguru.logger._core.handlers[0]._filter = _filter_logs
    logger = loguru.logger.opt(colors=True)
    logger.opt = partial(loguru.logger.opt, colors=True)
    logger.warn = logger.warning
    # logger.error = partial(logger.exception)

    if log_file:
        if not log_file.endswith(".log"):
            log_file = f"{log_file}.log"
        if not os.path.isabs(log_file):
            log_file = str((Settings.basic_settings.LOG_PATH / log_file).resolve())
        logger.add(log_file, colorize=False, filter=_filter_logs)

    return logger


logger = logging.getLogger(__name__)


class LoggerNameFilter(logging.Filter):
    def filter(self, record):
        # return record.name.startswith("{}_core") or record.name in "ERROR" or (
        #         record.name.startswith("uvicorn.error")
        #         and record.getMessage().startswith("Uvicorn running on")
        # )
        return True


def get_log_file(log_path: str, sub_dir: str):
    """
    sub_dir should contain a timestamp.
    """
    log_dir = os.path.join(log_path, sub_dir)
    # Here should be creating a new directory each time, so `exist_ok=False`
    os.makedirs(log_dir, exist_ok=False)
    return os.path.join(log_dir, f"{sub_dir}.log")


def get_config_dict(
        log_level: str, log_file_path: str, log_backup_count: int, log_max_bytes: int
) -> dict:
    # for windows, the path should be a raw string.
    log_file_path = (
        log_file_path.encode("unicode-escape").decode()
        if os.name == "nt"
        else log_file_path
    )
    log_level = log_level.upper()
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "formatter": {
                "format": (
                    "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
                )
            },
        },
        "filters": {
            "logger_name_filter": {
                "()": __name__ + ".LoggerNameFilter",
            },
        },
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
                # "stream": "ext://sys.stdout",
                # "filters": ["logger_name_filter"],
            },
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "formatter",
                "level": log_level,
                "filename": log_file_path,
                "mode": "a",
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "chatchat_core": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["stream_handler", "file_handler"],
        },
    }
    return config_dict


def get_timestamp_ms():
    t = time.time()
    return int(round(t * 1000))


async def get_search_results(params):
    try:
        url = "https://google.serper.dev/search"
        params["api_key"] = "e3d6ad661787a7318e1ccb5ed2ebdc431ce7b5c5"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                items = data.get("organic", [])
                results = []
                for item in items:
                    item["uuid"] = hashlib.md5(item["link"].encode()).hexdigest()
                    item["score"] = 0.00
                    results.append(item)
        return results
    except Exception as e:
        print("get search result failed: ", e)
        raise e


async def search(query, num=2, locale=''):
    params = {
        "q": query,
        "gl": "cn",
        "num": num,
        "hl": "zh-cn"
    }
    if locale:
        params["hl"] = locale

    try:
        search_results = await get_search_results(params=params)
        return search_results
    except Exception as e:
        print(f"search failed: {e}")
        raise e


async def fetch_url(session, url):
    try:
        async with session.get(url, ssl=False) as response:
            response.raise_for_stauts()
            response.encoding = 'utf-8'
            html = await response.text()
            return html
    except Exception as e:
        print(f"请求URL失败 {url} : {e}")
    return ""



async def html_to_markdown(html):
    try:
        converter = HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        markdown = converter.handle(html)
        return markdown
    except Exception as e:
        print(f"HTML 转换为 Md失败：{e}")
        return ""


async def fetch_markdown(session, url):
    try:
        html = await fetch_url(session, url)
        markdown = await html_to_markdown(html)

        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        return url, markdown

    except Exception as e:
        print(f"获取Md 失败 {url} ： {e}")
        return url, ""


async def batch_fetch_urls(urls):
    try:
        timeout = aiohttp.ClientTimeout(total=10, connect=-1)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [fetch_markdown(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            final_results = []
            for result in results:
                if isinstance(result, asyncio.TimeoutError):
                    continue
                elif isinstance(result, Exception):
                    pass
                else:
                    final_results.append(result)
            return final_results
    except Exception as e:
        print(f"批量获取url失败: {e}")
        return []


async def fetch_details(search_results):
    urls = [document.metadata['link'] for document in search_results if 'link' in document.metadata]
    try:
        details = await batch_fetch_urls(urls)
    except Exception as e:
        raise e

    content_maps = {url: content for url, content in details}

    for document in search_results:
        link = document.metadata['link']
        if link in content_maps:
            document.page_content = content_maps[link]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(search_results)
    return chunks


def md5(data: str):
    _md5 = hashlib.md5()
    _md5.update(data.encode('utf-8'))
    _hash = _md5.hexdigest()

    return _hash


def build_document(search_result):
    documents = []
    for result in search_result:
        if 'uuid' in result:
            uuid = result['uuid']
        else:
            uuid = md5(result['link'])
        text = result['snippet']

        document = Document(
            page_content=text,
            metadata={
                "uuid": uuid,
                "title": result["title"],
                "snippet": result["snippet"],
                "link": result["link"]
            },
        )
        documents.append(document)
    return documents
