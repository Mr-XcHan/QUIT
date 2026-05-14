from __future__ import annotations

from quit_agent.tools.device import select_torch_device


def test_auto_device_resolves_to_valid_torch_device():
    selection = select_torch_device("auto")

    assert selection.selected == "cpu" or selection.selected.startswith("cuda")
    assert selection.requested == "auto"
