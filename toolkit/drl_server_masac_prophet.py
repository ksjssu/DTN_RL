"""
Alias entrypoint for the PROPHET MA-SAC (CTDE SAC) DRL server.

The implementation lives in `toolkit/drl_server_rmappo_maxprop.py` and is
selected via request `protocol="masac_prophet_v1"` from Java RLBridgeReport.

This wrapper sets a more convenient default port (5010) to match existing
RLBridgeReport scenarios in this repo.
"""

from __future__ import annotations

import os


os.environ.setdefault("DRL_PORT", "5010")

from toolkit.drl_server_rmappo_maxprop import main


if __name__ == "__main__":
    main()

