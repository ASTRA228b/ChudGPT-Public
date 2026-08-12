"""Small conversational language-model package.

Model classes are intentionally not imported here. This keeps lightweight tools
such as data preparation usable without importing PyTorch during package setup.
Import ``ModelConfig`` and ``TransformerLM`` from ``chudlm.model`` directly.
"""
