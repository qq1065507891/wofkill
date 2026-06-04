"""Memory and review tests: cognition matrix, relation graph, reflection,
player profiles, review generation, cross-game retrieval, and store integration.
"""

import pytest

from werewolf_agent.cognition.belief import BeliefState, BeliefUpdater, PlayerBelief
from werewolf_agent.cognition.world_state import StructuredFact, StructuredWorldState
from werewolf_agent.memory.cognition_matrix import CognitionMatrix
from werewolf_agent.memory.profile import ProfileStore
from werewolf_agent.memory.reflection import ReflectionMemory
from werewolf_agent.memory.relation_graph import RelationGraph
from werewolf_agent.memory.review import ReviewGenerator
from werewolf_agent.memory.schemas import (
    CognitionMatrixEntry,
    CrossGameQuery,
    PlayerProfile,
    ReflectionEntry,
    RelationEvent,
    RelationType,
    ReviewJudgment,
    ReviewReport,
)
from werewolf_agent.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# CognitionMatrix
# ---------------------------------------------------------------------------

class TestCognitionMatrix:

    def test_initialize(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2", "p3"])
        assert set(cm.player_ids()) == {"p2", "p3"}
        assert "p1" not in cm.player_ids()

    def test_uniform_probabilities(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2"])
        entry = cm.get("p2")
        assert entry is not None
        probs = entry.role_probabilities
        assert len(probs) == 7
        for v in probs.values():
            assert abs(v - 1.0 / 7) < 0.01

    def test_custom_role_names(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2"], role_names=["villager", "werewolf"])
        entry = cm.get("p2")
        assert len(entry.role_probabilities) == 2

    def test_update_from_belief(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2", "p3"])

        belief = BeliefState(viewer_id="p1")
        belief.beliefs["p2"] = PlayerBelief(
            player_id="p2",
            role_probabilities={"werewolf": 0.7, "villager": 0.3},
            faction_lean="wolf_lean",
            trust=0.2,
            open_questions=["is_deep_hook"],
        )
        belief.beliefs["p3"] = PlayerBelief(
            player_id="p3",
            role_probabilities={"seer": 0.6, "villager": 0.4},
            faction_lean="good_lean",
            trust=0.8,
        )

        cm.update_from_belief(belief)
        p2 = cm.get("p2")
        assert p2.faction_read == "wolf_lean"
        assert p2.trust == 0.2
        assert p2.role_probabilities["werewolf"] == 0.7
        assert "is_deep_hook" in p2.open_questions

    def test_add_evidence(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2"])
        cm.add_evidence("p2", "day1_voted_player_03")
        entry = cm.get("p2")
        # MEM-07: bare str is wrapped into an EvidenceItem whose
        # claim equals the original string.
        assert len(entry.key_evidence) == 1
        assert entry.key_evidence[0].claim == "day1_voted_player_03"

    def test_add_open_question(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2"])
        cm.add_open_question("p2", "是否在倒钩")
        entry = cm.get("p2")
        assert "是否在倒钩" in entry.open_questions

    def test_to_dict_from_dict_roundtrip(self):
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2", "p3"])
        cm.add_evidence("p2", "ev1")
        cm.add_open_question("p3", "q1")

        data = cm.to_dict()
        cm2 = CognitionMatrix.from_dict(data)
        assert cm2.viewer_id == "p1"
        assert set(cm2.player_ids()) == {"p2", "p3"}
        # MEM-07: bare str is wrapped into an EvidenceItem.
        assert len(cm2.get("p2").key_evidence) == 1
        assert cm2.get("p2").key_evidence[0].claim == "ev1"
        assert "q1" in cm2.get("p3").open_questions

    def test_get_nonexistent(self):
        cm = CognitionMatrix("p1")
        assert cm.get("p99") is None

    def test_design_doc_format(self):
        """Verify the matrix matches design doc §10 JSON format."""
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2"])
        entry = cm.get("p2")
        d = entry.to_dict()
        assert "player_id" in d
        assert "role_probabilities" in d
        assert "faction_read" in d
        assert "trust" in d
        assert "key_evidence" in d
        assert "open_questions" in d


# ---------------------------------------------------------------------------
# RelationGraph
# ---------------------------------------------------------------------------

class TestRelationGraph:

    def test_add_and_count(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(
            predicate=RelationType.VOTED,
            source="p1", target="p2", day=1,
        ))
        assert rg.count() == 1

    def test_add_batch(self):
        rg = RelationGraph()
        events = [
            RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2", day=1),
            RelationEvent(predicate=RelationType.VOTED, source="p2", target="p3", day=1),
        ]
        rg.add_events(events)
        assert rg.count() == 2

    def test_query_by_predicate(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2"))
        rg.add_event(RelationEvent(predicate=RelationType.CLAIMED_ROLE, source="p1", value="seer"))
        voted = rg.by_predicate(RelationType.VOTED)
        assert len(voted) == 1
        assert voted[0].predicate == RelationType.VOTED

    def test_query_by_source(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2"))
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p2", target="p3"))
        p1_events = rg.by_source("p1")
        assert len(p1_events) == 1

    def test_query_by_target(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2"))
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p3", target="p2"))
        p2_events = rg.by_target("p2")
        assert len(p2_events) == 2

    def test_query_by_day(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2", day=1))
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p3", day=2))
        day1 = rg.by_day(1)
        assert len(day1) == 1

    def test_combined_query(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2", day=1))
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p3", day=2))
        rg.add_event(RelationEvent(predicate=RelationType.SPOKE_AGAINST, source="p1", target="p2", day=1))
        results = rg.query(predicate=RelationType.VOTED, source="p1", day=1)
        assert len(results) == 1
        assert results[0].target == "p2"

    def test_spoke_against(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(
            predicate=RelationType.SPOKE_AGAINST, source="p1", target="p2", day=1,
        ))
        assert rg.spoke_against("p1", "p2", 1)
        assert not rg.spoke_against("p1", "p2", 2)
        assert not rg.spoke_against("p2", "p1", 1)

    def test_voted_for(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2", day=1))
        rg.add_event(RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2", day=2))
        votes = rg.voted_for("p1", "p2")
        assert len(votes) == 2

    def test_claimed_roles(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(
            predicate=RelationType.CLAIMED_ROLE, source="p1", value="seer", day=1,
        ))
        rg.add_event(RelationEvent(
            predicate=RelationType.CLAIMED_ROLE, source="p2", value="witch", day=1,
        ))
        p1_claims = rg.claimed_roles("p1")
        assert len(p1_claims) == 1
        assert p1_claims[0].value == "seer"

    def test_defenses(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(
            predicate=RelationType.DEFENDED, source="p1", target="p2", day=1,
        ))
        defs = rg.defenses("p2")
        assert len(defs) == 1
        assert defs[0].source == "p1"

    def test_import_from_world_state_votes(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="vote", source_player="p1", target_player="p2", day=1))
        rg = RelationGraph()
        count = rg.import_from_world_state(ws)
        assert count == 1
        voted = rg.by_predicate(RelationType.VOTED)
        assert len(voted) == 1
        assert voted[0].source == "p1"

    def test_import_from_world_state_claims(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(
            fact_type="claimed_role", source_player="p1", value="seer", day=1,
        ))
        rg = RelationGraph()
        count = rg.import_from_world_state(ws)
        assert count == 1
        claims = rg.by_predicate(RelationType.CLAIMED_ROLE)
        assert len(claims) == 1

    def test_import_from_world_state_seer_check(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(
            fact_type="seer_check", source_player="p1", target_player="p2",
            value="good", night=1,
        ))
        rg = RelationGraph()
        count = rg.import_from_world_state(ws)
        assert count == 1
        night_claims = rg.by_predicate(RelationType.NIGHT_RESULT_CLAIMED)
        assert len(night_claims) == 1

    def test_import_speech_with_attack_keywords(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(
            fact_type="speech", source_player="p1", target_player="p2",
            value="我怀疑p2是狼坑", day=1,
        ))
        rg = RelationGraph()
        count = rg.import_from_world_state(ws)
        assert count >= 1
        attacks = rg.by_predicate(RelationType.SPOKE_AGAINST)
        assert len(attacks) >= 1

    def test_import_speech_with_defend_keywords(self):
        ws = StructuredWorldState()
        ws.append(StructuredFact(
            fact_type="speech", source_player="p1", target_player="p2",
            value="我保p2是好人", day=1,
        ))
        rg = RelationGraph()
        count = rg.import_from_world_state(ws)
        assert count >= 1
        defs = rg.by_predicate(RelationType.DEFENDED)
        assert len(defs) >= 1


# ---------------------------------------------------------------------------
# ReflectionMemory
# ---------------------------------------------------------------------------

class TestReflectionMemory:

    def _make_entry(self, entry_id="r1", player_id="p1", role="seer",
                    tags=None, text="learned something") -> ReflectionEntry:
        return ReflectionEntry(
            entry_id=entry_id,
            game_id="g1",
            player_id=player_id,
            role=role,
            faction_won=True,
            text=text,
            tags=tags or ["review", role],
            situation="endgame",
        )

    def test_store_and_get(self):
        mem = ReflectionMemory()
        entry = self._make_entry()
        mem.store(entry)
        assert mem.get("r1") is not None
        assert mem.get("r1").text == "learned something"

    def test_count(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1"))
        mem.store(self._make_entry("r2"))
        assert mem.count() == 2

    def test_delete(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1"))
        assert mem.delete("r1")
        assert mem.get("r1") is None
        assert not mem.delete("r1")

    def test_query_by_player(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1", player_id="p1"))
        mem.store(self._make_entry("r2", player_id="p2"))
        results = mem.by_player("p1")
        assert len(results) == 1

    def test_query_by_role(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1", role="seer"))
        mem.store(self._make_entry("r2", role="werewolf"))
        results = mem.by_role("seer")
        assert len(results) == 1

    def test_query_by_game(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1"))
        results = mem.by_game("g1")
        assert len(results) == 1
        results = mem.by_game("g999")
        assert len(results) == 0

    def test_cross_game_query_tags(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1", tags=["review", "seer", "win"]))
        mem.store(self._make_entry("r2", tags=["review", "werewolf", "loss"]))
        query = CrossGameQuery(tags=["seer"])
        results = mem.query(query)
        assert len(results) == 1
        assert results[0].entry_id == "r1"

    def test_cross_game_query_role(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1", role="seer"))
        mem.store(self._make_entry("r2", role="werewolf"))
        query = CrossGameQuery(role="werewolf")
        results = mem.query(query)
        assert len(results) == 1
        assert results[0].role == "werewolf"

    def test_cross_game_query_max_results(self):
        mem = ReflectionMemory()
        for i in range(10):
            mem.store(self._make_entry(f"r{i}", role="seer"))
        query = CrossGameQuery(role="seer", max_results=3)
        results = mem.query(query)
        assert len(results) == 3

    def test_cross_game_query_faction_won(self):
        mem = ReflectionMemory()
        mem.store(ReflectionEntry(
            entry_id="r1", game_id="g1", player_id="p1",
            role="seer", faction_won=True, text="won",
        ))
        mem.store(ReflectionEntry(
            entry_id="r2", game_id="g2", player_id="p1",
            role="seer", faction_won=False, text="lost",
        ))
        query = CrossGameQuery(player_id="p1", faction_won=False)
        results = mem.query(query)
        assert len(results) == 1
        assert results[0].faction_won is False

    def test_tag_index(self):
        mem = ReflectionMemory()
        mem.store(self._make_entry("r1", tags=["seer", "win"]))
        mem.store(self._make_entry("r2", tags=["seer", "loss"]))
        idx = mem.tag_index()
        assert idx["seer"] == 2
        assert idx["win"] == 1

    def test_roundtrip_dict(self):
        entry = self._make_entry()
        d = entry.to_dict()
        entry2 = ReflectionEntry.from_dict(d)
        assert entry2.entry_id == entry.entry_id
        assert entry2.text == entry.text
        assert entry2.tags == entry.tags

    # P0-M6: vector search support
    def test_query_with_vector_index(self):
        """Vector similarity surfaces semantically related entries."""
        from werewolf_agent.memory.vector_index import BagOfWordsVectorIndex

        mem = ReflectionMemory()
        mem.store(self._make_entry(
            "r1", text="上次站边预言家被冲爆,票投错了"
        ))
        mem.store(self._make_entry(
            "r2", text="学到的教训:不要轻信金水,要核对查验记录"
        ))
        mem.store(self._make_entry(
            "r3", text="其他话题的内容"
        ))

        idx = BagOfWordsVectorIndex()
        idx.add_text("r1", "站边 预言家 票 投错 冲爆")
        idx.add_text("r2", "金水 轻信 核对 查验")
        idx.add_text("r3", "其他 话题")
        idx.finalize()

        results = mem.query(
            CrossGameQuery(situation="站边 预言家 票型"),
            vector_index=idx,
        )
        # Both r1 and r2 should appear via vector similarity before r3
        entry_ids = [e.entry_id for e in results]
        assert "r1" in entry_ids
        # r1 is most similar to "站边 预言家 票型"
        assert entry_ids[0] == "r1"

    def test_reflection_falls_back_to_exact_match(self):
        """Without a vector index, exact-match behavior is preserved."""
        from werewolf_agent.memory.vector_index import BagOfWordsVectorIndex

        mem = ReflectionMemory()
        mem.store(ReflectionEntry(
            entry_id="r1", game_id="g1", player_id="p1", role="seer",
            faction_won=True, text="a", situation="endgame",
        ))
        mem.store(ReflectionEntry(
            entry_id="r2", game_id="g1", player_id="p1", role="seer",
            faction_won=True, text="b", situation="midgame",
        ))

        # No vector index provided → exact-match path only
        results = mem.query(CrossGameQuery(situation="endgame"))
        assert len(results) == 1
        assert results[0].entry_id == "r1"

        # Empty vector index → exact-match path only
        empty_idx = BagOfWordsVectorIndex()
        empty_idx.finalize()
        results = mem.query(
            CrossGameQuery(situation="endgame"),
            vector_index=empty_idx,
        )
        assert len(results) == 1
        assert results[0].entry_id == "r1"


# ---------------------------------------------------------------------------
# PlayerProfile
# ---------------------------------------------------------------------------

class TestPlayerProfile:

    def test_default_values(self):
        p = PlayerProfile(player_id="p1")
        assert p.logic == 0.5
        assert p.games_played == 0

    def test_win_rate_zero_games(self):
        p = PlayerProfile(player_id="p1")
        assert p.win_rate() == 0.0

    def test_win_rate_calculation(self):
        p = PlayerProfile(player_id="p1", games_played=10, wolf_wins=3, good_wins=4)
        assert abs(p.win_rate() - 0.7) < 0.01

    def test_apply_deltas(self):
        p = PlayerProfile(player_id="p1", logic=0.5)
        p.apply_deltas({"logic": 0.1, "credibility": -0.2})
        assert abs(p.logic - 0.6) < 0.01
        assert abs(p.credibility - 0.3) < 0.01

    def test_apply_deltas_clamped(self):
        p = PlayerProfile(player_id="p1", logic=0.95)
        p.apply_deltas({"logic": 0.1})
        assert p.logic == 1.0
        p.apply_deltas({"logic": -2.0})
        assert p.logic == 0.0

    def test_to_dict(self):
        p = PlayerProfile(player_id="p1")
        d = p.to_dict()
        assert d["player_id"] == "p1"
        assert "logic" in d
        assert "games_played" in d


class TestProfileStore:

    def test_get_or_create(self):
        store = ProfileStore()
        p = store.get_or_create("p1")
        assert p.player_id == "p1"
        p2 = store.get_or_create("p1")
        assert p is p2

    def test_get_nonexistent(self):
        store = ProfileStore()
        assert store.get("p99") is None

    def test_update_after_game_wolf_win(self):
        store = ProfileStore()
        p = store.update_after_game("p1", "werewolf", True, {"deception": 0.1}, "rev1")
        assert p.games_played == 1
        assert p.games_as_wolf == 1
        assert p.wolf_wins == 1
        assert p.good_wins == 0
        assert "rev1" in p.review_history

    def test_update_after_game_good_loss(self):
        store = ProfileStore()
        p = store.update_after_game("p1", "seer", False)
        assert p.games_played == 1
        assert p.games_as_good == 1
        assert p.good_wins == 0

    def test_multiple_games(self):
        store = ProfileStore()
        store.update_after_game("p1", "werewolf", True)
        store.update_after_game("p1", "seer", True)
        store.update_after_game("p1", "villager", False)
        p = store.get("p1")
        assert p.games_played == 3
        assert p.wolf_wins == 1
        assert p.good_wins == 1

    def test_top_by(self):
        store = ProfileStore()
        store.get_or_create("p1").logic = 0.9
        store.get_or_create("p2").logic = 0.3
        store.get_or_create("p3").logic = 0.7
        top = store.top_by("logic", limit=2)
        assert len(top) == 2
        assert top[0].player_id == "p1"
        assert top[1].player_id == "p3"

    def test_summary_empty(self):
        store = ProfileStore()
        s = store.summary()
        assert s["total_players"] == 0

    def test_summary_with_data(self):
        store = ProfileStore()
        store.update_after_game("p1", "werewolf", True)
        store.update_after_game("p2", "seer", False)
        s = store.summary()
        assert s["total_players"] == 2
        assert s["total_games_played"] == 2


# ---------------------------------------------------------------------------
# ReviewGenerator
# ---------------------------------------------------------------------------

class TestReviewGenerator:

    def _setup_matrix(self, viewer_id="p1") -> CognitionMatrix:
        cm = CognitionMatrix(viewer_id)
        cm.initialize([viewer_id, "p2", "p3", "p4"])
        return cm

    def test_basic_review(self):
        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth={"p2": "werewolf", "p3": "villager"},
        )
        assert report.game_id == "g1"
        assert report.player_id == "p1"
        assert report.role == "seer"
        assert report.faction_won is True

    def test_correct_judgment(self):
        cm = self._setup_matrix()
        # Set p2 as suspected werewolf
        entry = cm.get("p2")
        entry.role_probabilities = {"werewolf": 0.8, "villager": 0.2}
        entry.faction_read = "wolf_lean"

        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth={"p2": "werewolf", "p3": "villager"},
            cognition_matrix=cm,
        )
        correct = [j for j in report.key_judgments if j.judgment == "correct"]
        assert any(j.target_player == "p2" for j in correct)

    def test_incorrect_judgment_error_analysis(self):
        cm = self._setup_matrix()
        entry = cm.get("p2")
        entry.role_probabilities = {"villager": 0.8, "werewolf": 0.1}
        entry.faction_read = "good_lean"

        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=False,
            ground_truth={"p2": "werewolf", "p3": "villager"},
            cognition_matrix=cm,
        )
        assert len(report.error_analysis) > 0

    def test_ability_deltas(self):
        cm = self._setup_matrix()
        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True,
            ground_truth={"p2": "werewolf", "p3": "villager"},
            cognition_matrix=cm,
        )
        assert "credibility" in report.ability_deltas

    def test_wolf_deception_delta(self):
        cm = self._setup_matrix()
        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="werewolf",
            faction_won=True,
            ground_truth={"p2": "villager", "p3": "seer"},
            cognition_matrix=cm,
        )
        assert "deception" in report.ability_deltas
        assert report.ability_deltas["deception"] > 0

    def test_deception_analysis(self):
        rg = RelationGraph()
        rg.add_event(RelationEvent(
            predicate=RelationType.SPOKE_AGAINST, source="p5", target="p3", day=1,
        ))
        rg.add_event(RelationEvent(
            predicate=RelationType.VOTED, source="p1", target="p3", day=1,
        ))

        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=False,
            ground_truth={"p3": "seer", "p5": "werewolf"},
            relation_graph=rg,
        )
        # p1 voted for p3 (good), p5 (wolf) spoke against p3
        assert "p5" in report.deceived_by

    def test_improvement_suggestions(self):
        cm = self._setup_matrix()
        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=False,
            ground_truth={"p2": "villager", "p3": "werewolf"},
            cognition_matrix=cm,
        )
        assert len(report.improvement_suggestions) > 0

    def test_summary_generated(self):
        gen = ReviewGenerator()
        report = gen.generate(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True,
            ground_truth={"p2": "werewolf"},
        )
        assert report.summary != ""
        assert "seer" in report.summary


# ---------------------------------------------------------------------------
# MemoryStore integration
# ---------------------------------------------------------------------------

class TestMemoryStore:

    def _make_store_with_game(self) -> tuple[MemoryStore, dict[str, str]]:
        store = MemoryStore()
        ground_truth = {
            "p1": "seer", "p2": "werewolf", "p3": "villager",
            "p4": "witch", "p5": "werewolf",
        }
        for pid in ground_truth:
            store.init_matrix(pid, list(ground_truth.keys()))
        return store, ground_truth

    def test_init_matrix(self):
        store = MemoryStore()
        cm = store.init_matrix("p1", ["p1", "p2", "p3"])
        assert cm.viewer_id == "p1"
        assert store.get_matrix("p1") is not None

    def test_sync_matrix(self):
        store = MemoryStore()
        store.init_matrix("p1", ["p1", "p2"])
        belief = BeliefState(viewer_id="p1")
        belief.beliefs["p2"] = PlayerBelief(
            player_id="p2", role_probabilities={"werewolf": 0.9},
            faction_lean="wolf_lean", trust=0.1,
        )
        store.sync_matrix("p1", belief)
        cm = store.get_matrix("p1")
        assert cm.get("p2").faction_read == "wolf_lean"

    def test_add_relation(self):
        store = MemoryStore()
        store.add_relation(RelationEvent(
            predicate=RelationType.VOTED, source="p1", target="p2",
        ))
        assert store.relation_graph.count() == 1

    def test_import_world_state(self):
        store = MemoryStore()
        ws = StructuredWorldState()
        ws.append(StructuredFact(fact_type="vote", source_player="p1", target_player="p2", day=1))
        count = store.import_world_state(ws, day=1)
        assert count == 1

    def test_store_and_query_reflections(self):
        store = MemoryStore()
        store.store_reflection(ReflectionEntry(
            entry_id="r1", game_id="g1", player_id="p1",
            role="seer", faction_won=True,
            text="Don't trust emotional speeches",
            tags=["seer", "review", "win"],
        ))
        results = store.query_reflections(CrossGameQuery(player_id="p1"))
        assert len(results) == 1

    def test_generate_review(self):
        store, gt = self._make_store_with_game()
        report = store.generate_review(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth=gt,
        )
        assert report.game_id == "g1"
        assert report.player_id == "p1"
        assert len(report.key_judgments) > 0

    def test_review_updates_profile(self):
        store, gt = self._make_store_with_game()
        store.generate_review(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth=gt,
        )
        profile = store.get_profile("p1")
        assert profile is not None
        assert profile.games_played == 1
        assert profile.games_as_good == 1
        assert profile.good_wins == 1

    def test_review_stores_reflection(self):
        store, gt = self._make_store_with_game()
        store.generate_review(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth=gt,
        )
        reflections = store.reflections_by_player("p1")
        assert len(reflections) == 1
        assert "seer" in reflections[0].tags
        assert "review" in reflections[0].tags

    def test_generate_reviews_for_game(self):
        store, gt = self._make_store_with_game()
        reports = store.generate_reviews_for_game(
            game_id="g1",
            player_ids=list(gt.keys()),
            roles=gt,
            winning_faction="good",
            ground_truth=gt,
        )
        assert len(reports) == 5
        for r in reports:
            assert r.game_id == "g1"

    def test_cross_game_retrieval(self):
        store, gt = self._make_store_with_game()
        store.generate_review(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth=gt,
        )
        # Simulate new game setup
        store.reset_game_memory()
        store.init_matrix("p1", ["p1", "p2"])

        # Retrieve past experience
        experience = store.retrieve_experience("p1", role="seer")
        assert len(experience) == 1
        assert "seer" in experience[0].tags

    def test_reset_game_memory(self):
        store = MemoryStore()
        store.init_matrix("p1", ["p1", "p2"])
        store.add_relation(RelationEvent(
            predicate=RelationType.VOTED, source="p1", target="p2",
        ))
        store.reset_game_memory()
        assert store.get_matrix("p1") is None
        assert store.relation_graph.count() == 0

    def test_summary(self):
        store, gt = self._make_store_with_game()
        store.generate_review(
            game_id="g1", player_id="p1", role="seer",
            faction_won=True, ground_truth=gt,
        )
        s = store.summary()
        assert s["cognition_matrices"] == 5
        assert s["reflection_count"] == 1
        assert s["profile_count"] == 1

    def test_multiple_games_cross_game(self):
        """Simulate multiple games with cross-game retrieval."""
        store = MemoryStore()
        gt = {"p1": "seer", "p2": "werewolf", "p3": "villager"}

        # Game 1: seer wins
        for pid in gt:
            store.init_matrix(pid, list(gt.keys()))
        store.generate_review("g1", "p1", "seer", True, gt)
        store.reset_game_memory()

        # Game 2: seer loses
        for pid in gt:
            store.init_matrix(pid, list(gt.keys()))
        store.generate_review("g2", "p1", "seer", False, gt)
        store.reset_game_memory()

        # Both reflections should be retrievable
        experience = store.retrieve_experience("p1", role="seer", max_results=10)
        assert len(experience) == 2

        # Profile should have 2 games
        profile = store.get_profile("p1")
        assert profile.games_played == 2
        assert profile.good_wins == 1

    def test_profiles_persist_across_games(self):
        """Profiles and reflections survive game resets."""
        store = MemoryStore()
        gt = {"p1": "werewolf", "p2": "villager"}

        for pid in gt:
            store.init_matrix(pid, list(gt.keys()))
        store.generate_review("g1", "p1", "werewolf", True, gt)
        store.reset_game_memory()

        # Profile persists after reset
        assert store.get_profile("p1") is not None
        assert store.get_profile("p1").games_played == 1

        # Reflections persist after reset
        assert store.reflections.count() == 1


# ---------------------------------------------------------------------------
# MEM-05: situation/text redundancy in _store_review_reflection.
#
# The review-reflection path used to copy the scrubbed summary into
# BOTH ``entry.text`` AND ``entry.situation``. The two fields carried
# the same content, doubling the per-entry storage and (with
# _estimate_entry_tokens) the truncation cost. The fix keeps
# ``text`` as the primary reflection body and leaves ``situation``
# to carry only the game-context (day / role / game_id) snapshot,
# not the summary.
# ---------------------------------------------------------------------------


def test_review_reflection_no_text_situation_redundancy():
    """MEM-05: after a review, the resulting ReflectionEntry must not
    have the same content in both ``text`` and ``situation``.

    text = reflection body (summary / lessons / etc.)
    situation = structured game context, not a summary duplicate
    """
    from werewolf_agent.memory.schemas import ReviewReport

    store = MemoryStore()
    store.init_matrix("p1", ["p1", "p2"])
    report = ReviewReport(
        game_id="g_test_mem05",
        player_id="p1",
        role="seer",
        faction_won=True,
        summary="本局游戏的关键教训:谨慎金水,核对查验记录,不要轻信情绪化发言。",
    )
    store._store_review_reflection(report)
    reflections = store.reflections_by_player("p1")
    assert len(reflections) == 1
    entry = reflections[0]
    # The summary text must appear in `text`.
    assert "谨慎金水" in entry.text
    # And the situation field must NOT duplicate the summary.
    assert entry.situation != entry.text, (
        f"MEM-05: situation and text carry the same content; "
        f"text={entry.text!r} situation={entry.situation!r}"
    )
    # And the situation should NOT contain the full summary body.
    assert "谨慎金水" not in entry.situation, (
        f"MEM-05: situation duplicates the summary body; "
        f"situation={entry.situation!r}"
    )


# ---------------------------------------------------------------------------
# MEM-08: hybrid role with wolf master must be counted as wolf, not good.
#
# The legacy profile.update_after_game classified every non-"werewolf"
# role as good, which silently mis-classified hybrid (who wins with
# master's original faction). After the fix, callers pass an explicit
# ``faction`` argument and the profile increments games_as_wolf vs
# games_as_good accordingly.
# ---------------------------------------------------------------------------


def test_profile_hybrid_with_wolf_master_counted_as_wolf():
    """MEM-08: a hybrid whose master is on the wolf team must count
    in games_as_wolf (because hybrid wins with master's original
    faction) and not in games_as_good."""
    from werewolf_agent.memory.profile import ProfileStore

    store = ProfileStore()
    p = store.update_after_game(
        "p1", role="hybrid", faction_won=True, faction="werewolf",
    )
    assert p.games_played == 1
    assert p.games_as_wolf == 1, (
        f"MEM-08: hybrid with wolf master must count as wolf; "
        f"got games_as_wolf={p.games_as_wolf}, games_as_good={p.games_as_good}"
    )
    assert p.games_as_good == 0
    assert p.wolf_wins == 1


def test_profile_hybrid_with_good_master_counted_as_good():
    """MEM-08: hybrid with good master counts as good, not wolf."""
    from werewolf_agent.memory.profile import ProfileStore

    store = ProfileStore()
    p = store.update_after_game(
        "p1", role="hybrid", faction_won=True, faction="good",
    )
    assert p.games_played == 1
    assert p.games_as_good == 1
    assert p.games_as_wolf == 0
    assert p.good_wins == 1


def test_profile_hybrid_with_unknown_master_does_not_double_count():
    """MEM-08: if faction is unknown (e.g. master not yet determined),
    the game still counts in games_played but not in either
    games_as_wolf or games_as_good."""
    from werewolf_agent.memory.profile import ProfileStore

    store = ProfileStore()
    p = store.update_after_game(
        "p1", role="hybrid", faction_won=False, faction="unknown",
    )
    assert p.games_played == 1
    assert p.games_as_wolf == 0
    assert p.games_as_good == 0


# ---------------------------------------------------------------------------
# MEM-09: apply_deltas must log a warning when given an unknown attr key.
# Without it, typo'd deltas (e.g. ``win_rate`` instead of ``logic``) are
# silently dropped, and downstream code has no idea its input was
# rejected. A single warning is enough to flag the bug.
# ---------------------------------------------------------------------------


def test_apply_deltas_warns_on_unknown_attr(caplog):
    """MEM-09: unknown delta keys trigger a logger.warning."""
    import logging
    from werewolf_agent.memory.schemas import PlayerProfile

    p = PlayerProfile(player_id="p1", logic=0.5)
    with caplog.at_level(logging.WARNING, logger="werewolf_agent.memory.profile"):
        p.apply_deltas({"win_rate": 0.1})  # not in the whitelist

    # At least one warning recorded naming the unknown attr.
    matching = [r for r in caplog.records if "win_rate" in r.getMessage()]
    assert matching, (
        f"MEM-09: apply_deltas must warn on unknown attr keys; "
        f"got records: {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# MEM-10: ProfileStore.summary is observability-only and must be
# documented as such so future callers don't feed the aggregate into a
# player prompt (would leak the relative skill of every other player).
# ---------------------------------------------------------------------------


def test_profile_summary_documented_as_observability_only():
    """MEM-10: ProfileStore.summary's docstring must contain the word
    'observability' to flag it as not for prompt context."""
    from werewolf_agent.memory.profile import ProfileStore

    doc = (ProfileStore.summary.__doc__ or "")
    assert "observability" in doc.lower(), (
        f"MEM-10: ProfileStore.summary docstring must mark this as "
        f"observability-only; got doc={doc!r}"
    )


# ---------------------------------------------------------------------------
# MEM-14: lazy finalize with stale IDF / norms.
#
# The legacy BagOfWordsVectorIndex raised on add_text-after-finalize,
# which forced callers to either rebuild the whole index or skip
# late additions entirely. The post-fix behavior is to invalidate
# the cache so the next similarity() re-finalizes correctly.
# ---------------------------------------------------------------------------


def test_add_text_after_finalize_invalidates():
    """MEM-14: add_text after finalize() must not raise; it must
    invalidate the cached IDF/norms so the next similarity() call
    re-finalizes with the new docs included."""
    from werewolf_agent.memory.vector_index import BagOfWordsVectorIndex

    idx = BagOfWordsVectorIndex()
    idx.add_text("r1", "金水 轻信 冲爆")
    idx.finalize()
    # Sanity: cached stats are populated.
    assert idx._finalized is True
    assert idx._idf  # non-empty

    # Add a new doc post-finalize. With MEM-14 the call must NOT
    # raise; it must invalidate the cache.
    idx.add_text("r2", "站边 票型 预言家")

    # The index is no longer "finalized" — next similarity() will
    # re-finalize. The cached _idf is empty until that happens.
    assert idx._finalized is False
    assert idx._idf == {}
    assert idx._norms == {}

    # Calling similarity() now re-finalizes and surfaces the new
    # doc in the results. The query "站边 预言家" should rank r2
    # above r1 (r2 has both terms, r1 has neither).
    scores = idx.similarity("站边 预言家")
    assert "r2" in scores and "r1" in scores
    assert scores["r2"] > scores["r1"], (
        f"MEM-14: post-finalize add_text must be reflected in "
        f"similarity scores; got r1={scores['r1']} r2={scores['r2']}"
    )


# ---------------------------------------------------------------------------
# Boundary: structured data not vectors
# ---------------------------------------------------------------------------

class TestStructuredDataBoundary:

    def test_relation_events_are_queryable(self):
        """Vote chains, claims, attack/defense are structured and queryable."""
        rg = RelationGraph()
        rg.add_events([
            RelationEvent(predicate=RelationType.VOTED, source="p1", target="p2", day=1),
            RelationEvent(predicate=RelationType.VOTED, source="p3", target="p2", day=1),
            RelationEvent(predicate=RelationType.CLAIMED_ROLE, source="p4", value="seer", day=1),
            RelationEvent(predicate=RelationType.SPOKE_AGAINST, source="p5", target="p2", day=1),
        ])

        # Can query specific predicate types
        assert len(rg.by_predicate(RelationType.VOTED)) == 2
        assert len(rg.by_predicate(RelationType.CLAIMED_ROLE)) == 1
        assert len(rg.by_predicate(RelationType.SPOKE_AGAINST)) == 1

        # Can query by player
        p2_as_target = rg.by_target("p2")
        assert len(p2_as_target) == 3  # 2 votes + 1 spoke_against

    def test_cognition_matrix_is_json_serializable(self):
        """Short-term memory uses JSON, not vectors."""
        cm = CognitionMatrix("p1")
        cm.initialize(["p1", "p2"])
        d = cm.to_dict()
        import json
        serialized = json.dumps(d)
        assert "role_probabilities" in serialized
        assert "faction_read" in serialized

    def test_reflections_are_text_not_vectors(self):
        """Long-term reflections store text, not embeddings."""
        entry = ReflectionEntry(
            entry_id="r1", game_id="g1", player_id="p1",
            role="seer", faction_won=True,
            text="上次轻信情绪化发言导致误站边",
            tags=["seer", "review"],
        )
        mem = ReflectionMemory()
        mem.store(entry)
        retrieved = mem.get("r1")
        assert isinstance(retrieved.text, str)
        assert "情绪化" in retrieved.text
