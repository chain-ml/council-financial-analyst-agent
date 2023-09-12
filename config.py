import logging
from typing import List, Optional

import tiktoken
from llama_index.indices.base import BaseIndex
from transformers import AutoTokenizer
from tiktoken import Encoding

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import load_index_from_storage

import constants
from utils import check_index_files


class ChunkingTokenizer:
    """Tokenizer for chunking document data for creation of embeddings"""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)


class Config:
    """Configurations required for initializing the agent"""

    _llm_tokenizer: Optional[Encoding] = None

    def __init__(
        self,
        encoding_name: str,
        embedding_model_name: str,
    ):
        self._chunking_tokenizer = None
        self.encoding_name = encoding_name
        self.embedding_model_name = embedding_model_name

    def initialize(self) -> VectorStoreIndex:
        # Initialize tokenizer for text chunking
        self._chunking_tokenizer = ChunkingTokenizer(self.embedding_model_name)

        # Initialize text splitter
        self._text_splitter = TokenTextSplitter(
            chunk_size=constants.MAX_CHUNK_SIZE,
            chunk_overlap=constants.CHUNK_OVERLAP,
            tokenizer=self._chunking_tokenizer,
            separator="\n\n",
            backup_separators=["\n", " "],
        )

        # Initialize OpenAI LLM tokenizer
        self.llm_tokenizer = tiktoken.get_encoding(self.encoding_name)

        # Initialize vector index
        return self._init_index()

    def _init_index(self) -> VectorStoreIndex:
        node_parser = SimpleNodeParser(text_splitter=self._text_splitter)
        service_context = ServiceContext.from_defaults(
            embed_model=f"local:{self.embedding_model_name}", node_parser=node_parser
        )
        index_id = constants.COMPANY_NAME

        if check_index_files(constants.PERSIST_DIR):
            storage_context = StorageContext.from_defaults(
                persist_dir=constants.PERSIST_DIR
            )
            index = load_index_from_storage(
                storage_context=storage_context,
                service_context=service_context,
                index_id=index_id,
            )

            return index

        # If index does not exist, initialize index
        logging.info('message="initialize index started"')
        # Create index
        documents = SimpleDirectoryReader(constants.DOCUMENT_DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.set_index_id(index_id)
        # Save index to disk
        index.storage_context.persist(f"{constants.PERSIST_DIR}")
        logging.info('message="initialize index completed"')
        return index
