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

import pandas as pd
import sys
from fate.arch.dataframe import PandasReader
from fate.ml.glm.homo.lr.client import HomoLRClient
from fate.ml.glm.homo.lr.server import HomoLRServer


arbiter = ("arbiter", 10000)
guest = ("guest", 10000)
host = ("host", 9999)
name = "fed"


def create_ctx(local):
    from fate.arch import Context
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation.backends.standalone import StandaloneFederation
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, name, local, [guest, host, arbiter])
    )


if __name__ == "__main__":
    if sys.argv[1] == "guest":
        ctx = create_ctx(guest)
        df = pd.read_csv("../../../../../../../examples/data/breast_homo_guest.csv")
        df["sample_id"] = [i for i in range(len(df))]

        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="object")

        data = reader.to_frame(ctx, df)
        client = HomoLRClient(
            50,
            800,
            optimizer_param={"method": "adam", "penalty": "l1", "aplha": 0.1, "optimizer_para": {"lr": 0.1}},
            init_param={"method": "random", "fill_val": 1.0},
        )

        client.fit(ctx, data)

    elif sys.argv[1] == "host":
        ctx = create_ctx(host)
        df = pd.read_csv("../../../../../../../examples/data/breast_homo_host.csv")
        df["sample_id"] = [i for i in range(len(df))]

        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="object")

        data = reader.to_frame(ctx, df)
        client = HomoLRClient(
            50,
            800,
            optimizer_param={"method": "adam", "penalty": "l1", "aplha": 0.1, "optimizer_para": {"lr": 0.1}},
            init_param={"method": "random", "fill_val": 1.0},
        )

        client.fit(ctx, data)
    else:
        ctx = create_ctx(arbiter)
        server = HomoLRServer()
        server.fit(ctx)
