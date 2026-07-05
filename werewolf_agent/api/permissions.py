# -*- coding: utf-8 -*-
"""
功能描述：API 权限执行器，基于调用者角色与视图模式进行访问控制，拒绝操作自动写入审计日志。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：checker = PermissionChecker(); allowed = checker.check(caller_id, role, view, game_id)
"""

from __future__ import annotations

from datetime import datetime

from werewolf_agent.api.schemas import (
    AuditEvent,
    CallerRole,
    ViewMode,
)


class PermissionDenied(Exception):
    """Raised when access is denied."""

    def __init__(self, reason: str, audit: AuditEvent | None = None) -> None:
        super().__init__(reason)
        self.audit = audit
        self.reason = reason


class PermissionChecker:
    """Checks view mode access for API callers."""

    def __init__(self) -> None:
        self._audit_log: list[AuditEvent] = []

    def check(
        self,
        caller_id: str,
        caller_role: CallerRole,
        requested_view: ViewMode,
        game_id: str = "",
        endpoint: str = "",
        game_active: bool = True,
        target_player_id: str | None = None,
    ) -> ViewMode:
        """Check if caller can access the requested view mode.

        Returns the allowed view mode (may be downgraded).
        Raises PermissionDenied if access is completely denied.
        """
        # moderator and debugger can access everything
        if caller_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            return requested_view

        # Spectator can access public only
        if caller_role == CallerRole.SPECTATOR:
            if requested_view == ViewMode.PUBLIC:
                return ViewMode.PUBLIC
            # Downgrade to public for spectators requesting player_view
            if requested_view == ViewMode.PLAYER_VIEW:
                self._log(
                    caller_id=caller_id,
                    caller_role=caller_role,
                    requested_view=requested_view,
                    game_id=game_id,
                    endpoint=endpoint,
                    granted=True,
                    reason="Downgraded spectator player_view to public",
                )
                return ViewMode.PUBLIC
            # Deny moderator_full for spectators
            audit = self._log_denial(
                caller_id, caller_role, requested_view,
                game_id, endpoint, "Spectators cannot access moderator_full",
            )
            raise PermissionDenied(
                "Spectators cannot access moderator_full", audit
            )

        # Player agent restrictions
        if caller_role == CallerRole.PLAYER_AGENT:
            # During live play, moderator_full is forbidden
            if requested_view == ViewMode.MODERATOR_FULL:
                if game_active:
                    audit = self._log_denial(
                        caller_id, caller_role, requested_view,
                        game_id, endpoint,
                        "Player agents cannot access moderator_full during live play",
                    )
                    raise PermissionDenied(
                        "Player agents cannot access moderator_full during live play",
                        audit,
                    )
                # Post-game: still deny unless explicitly granted
                audit = self._log_denial(
                    caller_id, caller_role, requested_view,
                    game_id, endpoint,
                    "Player agents cannot access moderator_full",
                )
                raise PermissionDenied(
                    "Player agents cannot access moderator_full", audit
                )

            # player_view for own data is allowed
            if requested_view == ViewMode.PLAYER_VIEW:
                return ViewMode.PLAYER_VIEW

            # Public is always allowed
            if requested_view == ViewMode.PUBLIC:
                return ViewMode.PUBLIC

        raise PermissionDenied(f"Unknown role/view combination: {requested_view}")

    def check_private_state(
        self,
        caller_id: str,
        caller_role: CallerRole,
        target_player_id: str,
        game_id: str = "",
        endpoint: str = "",
    ) -> ViewMode:
        """Check access to a player's private state.

        Only the player themselves, moderator, or debugger can access.
        """
        if caller_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            return ViewMode.MODERATOR_FULL

        if caller_role == CallerRole.PLAYER_AGENT:
            if caller_id == target_player_id:
                return ViewMode.PLAYER_VIEW
            # Cannot access other players' private state
            audit = self._log_denial(
                caller_id, caller_role, ViewMode.PLAYER_VIEW,
                game_id, endpoint,
                f"Player agent {caller_id} cannot access private state of {target_player_id}",
            )
            raise PermissionDenied(
                f"Cannot access private state of player {target_player_id}", audit
            )

        # Spectator cannot access private state
        audit = self._log_denial(
            caller_id, caller_role, ViewMode.PLAYER_VIEW,
            game_id, endpoint,
            "Spectators cannot access private state",
        )
        raise PermissionDenied("Spectators cannot access private state", audit)

    def check_cognitive_diff(
        self,
        caller_id: str,
        caller_role: CallerRole,
        game_id: str = "",
        endpoint: str = "",
        game_active: bool = True,
    ) -> ViewMode:
        """Check access to cognitive diff view.

        Cognitive diff is moderator_full debug view only.
        Not available during live play for player agents.
        """
        if caller_role in (CallerRole.MODERATOR, CallerRole.DEBUGGER):
            return ViewMode.MODERATOR_FULL

        audit = self._log_denial(
            caller_id, caller_role, ViewMode.MODERATOR_FULL,
            game_id, endpoint,
            "Cognitive diff view requires moderator or debugger role",
        )
        raise PermissionDenied(
            "Cognitive diff view requires moderator or debugger role", audit
        )

    def _log_denial(
        self,
        caller_id: str,
        caller_role: CallerRole,
        requested_view: ViewMode,
        game_id: str,
        endpoint: str,
        reason: str,
    ) -> AuditEvent:
        return self._log(
            caller_id=caller_id,
            caller_role=caller_role,
            requested_view=requested_view,
            game_id=game_id,
            endpoint=endpoint,
            granted=False,
            reason=reason,
        )

    def _log(
        self,
        caller_id: str,
        caller_role: CallerRole,
        requested_view: ViewMode,
        game_id: str,
        endpoint: str,
        granted: bool,
        reason: str,
    ) -> AuditEvent:
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            caller_id=caller_id,
            caller_role=caller_role,
            requested_view=requested_view,
            game_id=game_id,
            endpoint=endpoint,
            granted=granted,
            reason=reason,
        )
        self._audit_log.append(event)
        return event

    def audit_log(self) -> list[AuditEvent]:
        return list(self._audit_log)

    def denials(self) -> list[AuditEvent]:
        return [e for e in self._audit_log if not e.granted]

    def clear_audit_log(self) -> None:
        self._audit_log.clear()
