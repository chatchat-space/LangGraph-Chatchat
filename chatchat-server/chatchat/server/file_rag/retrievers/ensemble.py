from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers.bm25 import default_preprocessing_func
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService

import time
import rich


class BM25RetrieverNew(BM25Retriever):
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        print(" 🚗 this is BM25RetrieverNew from_texts")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        start_time = time.time()
        texts_processed = [preprocess_func(t) for t in texts]
        print(f" 🚗 BM25RetrieverNew from_texts texts_processed in {time.time() - start_time:.4f} seconds")

        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)

        start_time = time.time()
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        print(f" 🚗 BM25RetrieverNew from_texts docs processed 2 in {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        cls_finally = cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )
        print(f" 🚗 BM25RetrieverNew from_texts cls_finally in {time.time() - start_time:.4f} seconds")
        return cls_finally

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        print(" 🚢 this is BM25RetrieverNew from_documents")

        start_time = time.time()
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        print(f" 🚢 BM25RetrieverNew from_documents texts in {time.time() - start_time:.4f} seconds")

        text_demo = texts[0]
        print(f" 🚢 texts type: {type(texts)}, text_demo type: {type(text_demo)}, text_demo:")
        rich.print(text_demo)

        metadata_demo = metadatas[0]
        print(f" 🚢 metadatas type: {type(metadatas)}, metadata_demo type: {type(metadata_demo)}, metadata_demo:")
        rich.print(metadata_demo)

        cls_from_texts = cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )
        return cls_from_texts


class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @staticmethod
    def from_vectorstore(vectorstore: VectorStore,
                         top_k: int,
                         score_threshold: int | float):

        print(f" ✅ ✅ vectorstore:")
        rich.print(vectorstore)

        start_time = time.time()
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
        print(f" ✅ ✅ faiss_retriever created in {time.time() - start_time:.4f} seconds")
        # TODO: 换个不用torch的实现方式
        # from cutword.cutword import Cutter
        import jieba

        # cutter = Cutter()

        start_time = time.time()
        docs = list(vectorstore.docstore._dict.values())
        print(f" ✅ ✅ found {len(docs)}")
        print(f" ✅ ✅ list(vectorstore.docstore._dict.values()) in {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        bm25_retriever = BM25RetrieverNew.from_documents(
            docs,
            preprocess_func=jieba.lcut_for_search,
        )
        print(f" ✅ ✅ bm25_retriever created in {time.time() - start_time:.4f} seconds")
        bm25_retriever.k = top_k

        start_time = time.time()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        print(f" ✅ ✅ ensemble_retriever created in {time.time() - start_time:.4f} seconds")

        return EnsembleRetrieverService(retriever=ensemble_retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)[: self.top_k]
