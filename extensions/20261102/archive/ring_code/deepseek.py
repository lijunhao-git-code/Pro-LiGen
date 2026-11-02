"""Portable credential adapter when the original server's provider is absent.

The existing server provider retains precedence in the frozen service.settings().
Supply credentials via environment variables; never copy private configuration files.
"""
import os


def _get_base_url():return os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com')


def _get_api_key():
    key=os.environ.get('DEEPSEEK_API_KEY')
    if not key:raise RuntimeError('Set DEEPSEEK_API_KEY in the process environment; never write it into the experiment package.')
    return key
