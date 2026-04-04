"""
Environment implementation used by the HTTP server.

Logic lives in :class:`helpdesk_env.environment.HelpdeskEnv`; this module is a
stable import path for OpenEnv-style layouts (``server/my_environment.py``).
"""

from ..environment import HelpdeskEnv

__all__ = ["HelpdeskEnv"]
