import requests
import json
from typing import Tuple, Dict
from defines import *


def blockchain_request(
    chain_root: str, interface: str, data: Dict, param: bool = False
):
    headers = {"Content-Type": "application/json"}
    rest_url = f"{chain_root}/{interface}"

    if param:
        response = requests.post(rest_url, params=data, headers=headers)
    else:
        response = requests.post(rest_url, data=json.dumps(data), headers=headers)
    return response.json()
