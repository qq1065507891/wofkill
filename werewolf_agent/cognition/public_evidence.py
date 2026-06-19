"""Public evidence index shared by cognition and evaluation.

The index stores public claim/vote anchors derived from visible structured
facts. It never reads ground truth and can be snapshotted by incremental
runtime cognition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact


_VOTE_SIGNAL_DELTA = {
    "checked_wolf": 0.04,
    "gold_water": -0.04,
    "seer_claimant": -0.03,
    "public_suspect": 0.03,
}


@dataclass(frozen=True)
class EvidenceRef:
    source_player: str
    target_player: str
    fact_type: str
    value: str = ""
    day: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VoteRef:
    voter: str
    target: str
    day: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicEvidenceIndex:
    checked_wolves: dict[str, list[EvidenceRef]] = field(default_factory=dict)
    gold_water: dict[str, list[EvidenceRef]] = field(default_factory=dict)
    seer_claimants: dict[str, list[EvidenceRef]] = field(default_factory=dict)
    public_suspects: dict[str, list[EvidenceRef]] = field(default_factory=dict)
    votes: list[VoteRef] = field(default_factory=list)

    def observe(self, fact: StructuredFact) -> None:
        source = fact.source_player or ""
        target = fact.target_player or ""
        value = fact.value or ""
        if fact.fact_type == "seer_check_claim" and target:
            if is_wolf_result(value):
                self._add(self.checked_wolves, target, fact)
            elif is_good_result(value):
                self._add(self.gold_water, target, fact)
        elif fact.fact_type == "claimed_good" and target:
            self._add(self.gold_water, target, fact)
        elif fact.fact_type == "claimed_role" and value.lower() == "seer" and source:
            self._add(self.seer_claimants, source, fact)
        elif fact.fact_type == "claimed_suspect" and target:
            self._add(self.public_suspects, target, fact)
        elif fact.fact_type == "vote" and source and target:
            self.votes.append(VoteRef(voter=source, target=target, day=fact.day))

    def vote_delta(self, vote: StructuredFact) -> float:
        target = vote.target_player or ""
        if not target:
            return 0.0
        delta = 0.0
        if target in self.checked_wolves:
            delta += _VOTE_SIGNAL_DELTA["checked_wolf"]
        if target in self.gold_water:
            delta += _VOTE_SIGNAL_DELTA["gold_water"]
        if target in self.seer_claimants:
            delta += _VOTE_SIGNAL_DELTA["seer_claimant"]
        if target in self.public_suspects:
            delta += _VOTE_SIGNAL_DELTA["public_suspect"]
        return delta

    def vote_targets(self, voter: str) -> set[str]:
        return {v.target for v in self.votes if v.voter == voter and v.target}

    def supports_reference(self, player_id: str, concept: str) -> bool:
        if concept == "black_check":
            return player_id in self.checked_wolves
        if concept == "gold_water":
            return player_id in self.gold_water
        if concept == "seer_claim":
            return player_id in self.seer_claimants
        if concept == "public_suspect":
            return player_id in self.public_suspects
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "checked_wolves": self._refs_to_json(self.checked_wolves),
            "gold_water": self._refs_to_json(self.gold_water),
            "seer_claimants": self._refs_to_json(self.seer_claimants),
            "public_suspects": self._refs_to_json(self.public_suspects),
            "votes": [vote.to_json_dict() for vote in self.votes],
        }

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> "PublicEvidenceIndex":
        return cls(
            checked_wolves=cls._refs_from_json(snap.get("checked_wolves", {})),
            gold_water=cls._refs_from_json(snap.get("gold_water", {})),
            seer_claimants=cls._refs_from_json(snap.get("seer_claimants", {})),
            public_suspects=cls._refs_from_json(snap.get("public_suspects", {})),
            votes=[
                VoteRef(
                    voter=str(v.get("voter", "")),
                    target=str(v.get("target", "")),
                    day=int(v.get("day", 0) or 0),
                )
                for v in snap.get("votes", [])
            ],
        )

    @staticmethod
    def _add(bucket: dict[str, list[EvidenceRef]], key: str, fact: StructuredFact) -> None:
        ref = EvidenceRef(
            source_player=fact.source_player or "",
            target_player=fact.target_player or key,
            fact_type=fact.fact_type,
            value=fact.value or "",
            day=fact.day,
        )
        refs = bucket.setdefault(key, [])
        if ref not in refs:
            refs.append(ref)

    @staticmethod
    def _refs_to_json(bucket: dict[str, list[EvidenceRef]]) -> dict[str, list[dict[str, Any]]]:
        return {key: [ref.to_json_dict() for ref in refs] for key, refs in bucket.items()}

    @staticmethod
    def _refs_from_json(data: dict[str, Any]) -> dict[str, list[EvidenceRef]]:
        return {
            str(key): [
                EvidenceRef(
                    source_player=str(item.get("source_player", "")),
                    target_player=str(item.get("target_player", "")),
                    fact_type=str(item.get("fact_type", "")),
                    value=str(item.get("value", "")),
                    day=int(item.get("day", 0) or 0),
                )
                for item in refs
                if isinstance(item, dict)
            ]
            for key, refs in data.items()
            if isinstance(refs, list)
        }


def is_wolf_result(value: str) -> bool:
    v = value or ""
    return "wolf" in v.lower() or "狼" in v or "查杀" in v


def is_good_result(value: str) -> bool:
    v = value or ""
    return "good" in v.lower() or "好人" in v or "金水" in v
