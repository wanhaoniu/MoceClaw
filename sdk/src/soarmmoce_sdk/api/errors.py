# -*- coding: utf-8 -*-

class SoarmMoceError(Exception):
    """Base SDK error."""


class ConnectionError(SoarmMoceError):
    pass


class ProtocolError(SoarmMoceError):
    pass


class TimeoutError(SoarmMoceError):
    pass


class IKError(SoarmMoceError):
    pass


class LimitError(SoarmMoceError):
    pass


class CapabilityError(SoarmMoceError):
    """Raised when a requested capability is unsupported by current transport/backend."""

    pass


class PermissionError(SoarmMoceError):
    """Raised when a control command is blocked by SDK permission policy."""

    pass
