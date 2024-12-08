# Web scrapping to extract data from UniprotKB
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from Bio import SeqIO
from io import StringIO
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import time
import gzip

# Define the UniProt query URL for reviewed transferase proteins in TSV format
url = 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Cid%2Clength%2Csequence%2Cec%2Cdate_created&format=tsv&query=%28ec%3A*%29+AND+%28reviewed%3Atrue%29'
# Set up retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

def download_tsv_with_retries(url, session, retries=5):
    for i in range(retries):
        try:
            response = session.get(url)
            response.raise_for_status()  # Check if the request was successful
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
               time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
# Download the data with retry logic
response = download_tsv_with_retries(url, http)

# Check if the data is compressed and handle accordingly
if response.headers.get('Content-Encoding') == 'gzip':
    tsv_data = gzip.decompress(response.content).decode('utf-8')
else:
    tsv_data = response.content.decode('utf-8')

# Read the data into a pandas DataFrame
#tsv_data = response.text
#data = pd.read_csv(pd.compat.StringIO(tsv_data), sep='\t')
data = pd.read_csv(StringIO(tsv_data), sep='\t')

# Save the DataFrame to an Excel file
#data.to_csv("uniprot_enzyme_unseen_data.csv", index=False)
