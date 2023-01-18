from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum

class ItemEnum(Enum):
   alexnet = "alexnet"
   resnet = "resnet"
   lenet = "lenet"

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/query_items")
def read_item(item_id: int):
   return {"item_id": item_id}

@app.get("/test_enum")
def read_item(item_id: int):
    return {"item_id": item_id}

