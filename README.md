# project calmzeus

Exploration for project [peacefulares](https://github.com/lo-b/peacefulares).

## Development

### Jupyter Ascending (optional)

> _"Jupyter Ascending lets you edit Jupyter notebooks from your favorite editor,
> then instantly sync and execute that code in the Jupyter notebook running in
> your browser."_ -- [source]()

#### Setup:

In short: install 'jupyter_ascending' package (defined as dev dependency in
pyproject) and run the following commands to setup/config server:

```zsh
poetry run python -m jupyter nbextension    install jupyter_ascending --sys-prefix --py
```

```zsh
poetry run python -m jupyter nbextension     enable jupyter_ascending --sys-prefix --py
```

```zsh
poetry run python -m jupyter serverextension enable jupyter_ascending --sys-prefix --py
```
