"""
Agent implementations for external system integrations.

This package contains agents that interface with external services:
- VectorizationAgent: OpenAI API integration for sentence embeddings
"""

from agents.vectorization_agent import VectorizationAgent

__all__ = ["VectorizationAgent"]
