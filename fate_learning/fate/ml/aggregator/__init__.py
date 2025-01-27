#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient, PlainTextAggregatorServer
from fate.ml.aggregator.secure_aggregator import SecureAggregatorClient, SecureAggregatorServer
import enum


class AggregatorType(enum.Enum):
    PLAINTEXT = "plaintext"
    SECURE_AGGREGATE = "secure_aggregate"


aggregator_map = {
    AggregatorType.PLAINTEXT.value: (PlainTextAggregatorClient, PlainTextAggregatorServer),
    AggregatorType.SECURE_AGGREGATE.value: (SecureAggregatorClient, SecureAggregatorServer),
}

from fate.ml.aggregator.aggregator_wrapper import AggregatorClientWrapper, AggregatorServerWrapper

__all__ = [
    "PlainTextAggregatorClient",
    "PlainTextAggregatorServer",
    "SecureAggregatorClient",
    "SecureAggregatorServer",
    "AggregatorServerWrapper",
    "AggregatorClientWrapper",
]
