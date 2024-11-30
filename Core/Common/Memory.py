#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/06 17:01
@Author  : 🛞
@File    : memory.py
"""
from collections import defaultdict
from typing import DefaultDict, Iterable, Set

from pydantic import BaseModel, Field, SerializeAsAny

from Core.Common.Constants import IGNORED_MESSAGE_ID
from Core.Schema.Message import Message
from Core.Common.Utils import any_to_str, any_to_str_set


class Memory(BaseModel):
    """The most basic memory: super-memory"""

    storage: list[SerializeAsAny[Message]] = []
    ignore_id: bool = False

    def add(self, message: Message):
        """Add a new message to storage, while updating the index"""
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID
        if message in self.storage:
            return
        self.storage.append(message)


    def add_batch(self, messages: Iterable[Message]):
        for message in messages:
            self.add(message)


    def get_by_content(self, content: str) -> list[Message]:
        """Return all messages containing a specified content"""
        return [message for message in self.storage if content in message.content]

    def delete_newest(self) -> "Message":
        """delete the newest message from the storage"""
        if len(self.storage) > 0:
            newest_msg = self.storage.pop()
           
        else:
            newest_msg = None
        return newest_msg

    def delete(self, message: Message):
        """Delete the specified message from storage, while updating the index"""
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID
        self.storage.remove(message)
      

    def clear(self):
        """Clear storage and index"""
        self.storage = []

    def count(self) -> int:
        """Return the number of messages in storage"""
        return len(self.storage)

    def try_remember(self, keyword: str) -> list[Message]:
        """Try to recall all messages containing a specified keyword"""
        return [message for message in self.storage if keyword in message.content]

    def get(self, k=0) -> list[Message]:
        """Return the most recent k memories, return all when k=0"""
        return self.storage[-k:]

    def find_news(self, observed: list[Message], k=0) -> list[Message]:
        """find news (previously unseen messages) from the most recent k memories, from all memories when k=0"""
        already_observed = self.get(k)
        news: list[Message] = []
        for i in observed:
            if i in already_observed:
                continue
            news.append(i)
        return news

    def get_by_action(self, action) -> list[Message]:
        """Return all messages triggered by a specified Action"""
        index = any_to_str(action)
        return self.index[index]

    def get_by_actions(self, actions: Set) -> list[Message]:
        """Return all messages triggered by specified Actions"""
        rsp = []
        indices = any_to_str_set(actions)
        for action in indices:
            if action not in self.index:
                continue
            rsp += self.index[action]
        return rsp