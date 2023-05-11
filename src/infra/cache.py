from collections import OrderedDict
from typing import Optional, Type

from src.tictactoe.t3_tree import Node


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()  # type: OrderedDict[str, Type["Node"]]
        self.capacity = capacity

    def get(self, key: str) -> Optional[Type["Node"]]:
        if key in self.cache:
            # Move the recently accessed element to the end of the cache
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Type["Node"]) -> None:
        if key in self.cache:
            # If the key already exists, move it to the end
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove the least recently used element
            self.cache.popitem(last=False)

    def __str__(self):
        return f"LRUCache({self.cache})"
