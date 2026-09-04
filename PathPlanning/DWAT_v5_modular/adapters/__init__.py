"""Adapters that turn external environment datasets into v5 scenarios.

Each adapter reads one dataset from EnvData/ and writes two artefacts: a
scenario JSON under configs/scenarios/ and a track bundle .npz. Nothing
downstream needs to know which dataset a scenario came from.

The code lives here rather than beside the data because EnvData/ is git-ignored;
only the generated data is written back there.
"""
