import os
from google.cloud import storage

if __name__ == "__main__" : 

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    client =  storage.Client() 
    bucket = client.bucket()

    blob = bucket.blob()

