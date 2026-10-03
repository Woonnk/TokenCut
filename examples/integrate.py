"""Run locally with: python examples/integrate.py. No model call or API key needed."""

import json
from pathlib import Path

from tokencut import ContextPacket, TokenCounter, optimize

packet_path = Path(__file__).with_name("context.json")
packet = ContextPacket.from_dict(json.loads(packet_path.read_text(encoding="utf-8")))
result = optimize(packet, counter=TokenCounter("estimate"))
print(result.render())
print(json.dumps(result.report, indent=2))

# Pass result.render() as retrieved data at its existing trust level in your agent.
# Keep result.report out of model context. It is for local diagnostics only.
