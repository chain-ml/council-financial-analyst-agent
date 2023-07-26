# Data information
DOCUMENT_DATA_DIR = "./document_data"
PERSIST_DIR = "./storage"
MARKET_DATA_DIR = "./market_data"
COMPANY_NAME = "Microsoft"
COMPANY_TICKER = "MSFT"

# LLM Model Selection
DOC_AND_GOOGLE_RETRIEVAL_LLM = 'gpt-3.5-turbo'
PANDAS_LLM = 'gpt-3.5-turbo'
CONTROLLER_LLM = 'gpt-4'

# Document retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODING_NAME = "cl100k_base"
MAX_CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
CONTEXT_TOKEN_LIMIT = 3000
NUM_RETRIEVED_DOCUMENTS = 50
