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
