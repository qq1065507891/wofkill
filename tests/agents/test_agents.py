"""Backward-compatibility re-imports from domain-focused test files.

All test classes and mocks have been moved to:
- tests/agents/test_schemas.py
- tests/agents/test_visibility.py
- tests/agents/test_persona_router.py
- tests/agents/test_player_agent.py
- tests/agents/test_model_router.py
- tests/agents/test_judge_agent.py
"""

# Schemas
from tests.agents.test_schemas import (  # noqa: F401
    TestPlayerActionSchema,
    TestJudgeBroadcastSchema,
    TestDefaultActionValidator,
)

# Visibility
from tests.agents.test_visibility import TestVisibilityBoundaries  # noqa: F401

# Persona router
from tests.agents.test_persona_router import TestPersonaRouter  # noqa: F401

# Player agent (all test classes)
from tests.agents.test_player_agent import (  # noqa: F401
    TestPlayerAgentRetryFallback,
    TestMandatoryVote,
    TestSpeechQualityAndWolfAssignments,
    TestWitchNoPoisonMustExplain,
    TestPlainTextRejection,
    TestProviderCapabilityFailure,
    TestStructuredOutputMetadata,
    TestSpeechMustAnswerVisibleContradictionAlert,
)

# Player agent mocks
from tests.agents.test_player_agent import (  # noqa: F401
    _FailProvider,
    _JsonProvider,
    _SequenceJsonProvider,
    _FakeHttpResponse,
    _FakeHttpClient,
    ToolAwareProvider,
    EmptyFailureRouter,
    LegacyProvider,
    ToolProbeProvider,
    TextProbeProvider,
    TextOnlyProvider,
    TextJsonProvider,
    NoToolProvider,
)

# Model router
from tests.agents.test_model_router import TestModelRouter  # noqa: F401

# Judge agent
from tests.agents.test_judge_agent import (  # noqa: F401
    TestJudgeAgent,
    TestAgentIntegration,
)
