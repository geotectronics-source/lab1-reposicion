"""Permite invocar el pipeline como `python -m reposicion <etapa>`.

Es la forma que usan `dvc.yaml` y los DAGs en vez del script instalado
`reposicion`: no depende de que el directorio de scripts del entorno este en
el PATH, que es justo lo que cambia entre Windows, Linux y el contenedor de
Airflow.
"""

from reposicion.cli import app

app()
