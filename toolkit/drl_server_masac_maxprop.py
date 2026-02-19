"""
Alias entrypoint for the MaxProp++ DRL server.

This repository historically used the filename `drl_server_rmappo_maxprop.py`.
The mp_v1 path now runs MA-SAC (CTDE SAC), but other tooling may still import
the original module. This file provides a stable MA-SAC-named entrypoint
without breaking backward compatibility.
"""

from __future__ import annotations

from toolkit.drl_server_rmappo_maxprop import main


if __name__ == "__main__":
    main()

