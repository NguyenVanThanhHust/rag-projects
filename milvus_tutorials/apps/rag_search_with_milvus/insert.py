import sys
import os
import ssl
import certifi
from glob import glob
from tqdm import tqdm

from encoder import emb_text
from milvus_utils import get_milvus_client, create_collection

from dotenv import load_dotenv

load_dotenv()