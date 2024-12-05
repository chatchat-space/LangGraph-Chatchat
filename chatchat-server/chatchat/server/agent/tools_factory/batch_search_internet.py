import aiohttp
import asyncio
import re
import hashlib

from pydantic import Field

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chatchat.server.agent.tools_factory.tools_registry import regist_tool
from chatchat.server.utils import get_tool_config


async def get_search_results(params):
    try:
        config = get_tool_config("search_internet")["search_engine_config"]["google"]
        url = config["google_search_url"]
        params["api_key"] = config["google_key"]

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
    from html2text import HTML2Text
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


def md5(data: str):
    _md5 = hashlib.md5()
    _md5.update(data.encode('utf-8'))
    _hash = _md5.hexdigest()

    return _hash


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


@regist_tool(title="互联网批量搜索")
async def serperV2(query: str = Field(description="The search query title")):
    """
    useful for when you need to search the internet for information
    translate user question to serperV2 Required questions that can be evaluated by serperV2
    """
    response = await search(query)
    result = await fetch_details(build_document(response))
    return result
