#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Core.Community.LeidenCommunity import (
 LeidenCommunity
)

from Core.Community.RaptorClustering import (RaptorClustering)


_ = (
    LeidenCommunity,
    RaptorClustering
)  # Avoid pre-commit error
