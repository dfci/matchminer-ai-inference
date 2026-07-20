# Config Loading

Most users only need these helpers:

- Use `load_default_preset()` to start from the built-in default settings and
  change a few values in Python.
- Use `load_config()` to load your own YAML config file.
- Use `load_preset()` when loading a named preset packaged with
  `matchminer-ai`.

::: matchminer_ai.config
    options:
      members:
        - load_config
        - load_default_preset
        - load_preset
