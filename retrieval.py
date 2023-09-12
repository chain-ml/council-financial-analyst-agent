from llama_index.indices.vector_store import VectorIndexRetriever

import constants
from config import Config

from typing import List
from llama_index.schema import NodeWithScore


class Retriever:
    """Class to retrieve text chunks from Llama Index and create context for LLM."""
    def __init__(self, config: Config, retriever: VectorIndexRetriever):
        self.llm_tokenizer = config.llm_tokenizer
        self.retriever = retriever

    def retrieve_docs(self, query) -> str:
        """End-to-end function to retrieve most similar nodes and build the context"""
        nodes = self.retriever.retrieve(query)
        docs = self._extract_text(nodes)
        context = self._build_context(docs)

        return context

    @staticmethod
    def _extract_text(nodes: List[NodeWithScore]) -> List[str]:
        """Function to extract the text from the retrieved nodes"""
        return [node.node.text for node in nodes]

    def _build_context(self, docs: List[str]) -> str:
        """Function to build context for LLM by separating text chunks into paragraphs"""
        context = ""
        num_tokens = 0
        for doc in docs:
            doc += "\n\n"
            num_tokens += len(self.llm_tokenizer.encode(doc))
            if num_tokens <= constants.CONTEXT_TOKEN_LIMIT:
                context += doc
            else:
                break

        return context
