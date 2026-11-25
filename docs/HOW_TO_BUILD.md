# Сборка документации

Из корня репозитория установите пакет и инструменты документации в выбранное окружение:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
```

Для проверки примеров нейросетей установите PyTorch через `python -m pip install -e '.[torch]'`.
CUDA не обязательна.

На всех платформах сборка запускается одной командой:

```bash
python docs/build.py
```

В Windows также можно запустить `docs\rebuild-docs.bat`, а из каталога `docs`
на macOS/Linux — `make html PYTHON=python`.
Все эти команды собирают текущую версию в `docs/0.3.0/` и обновляют перенаправление
`docs/index.html` только после успешной сборки. Архив `docs/_archive/` не пересобирается.

Версия и список ссылок переключателя задаются переменными `DOC_VERSION` и
`DOC_VERSIONS`; по умолчанию это `0.3.0` и `0.3.0,0.2.7`.
Сборка использует Intersphinx-инвентарь scikit-learn для ссылок на унаследованные методы.
При первом запуске нужен доступ к `https://scikit-learn.org/stable/objects.inv`.

Для проверки без изменения публикуемого сайта используйте отдельный каталог вывода:

```bash
python -m sphinx -E -a -W --keep-going -b html -d /tmp/thefittest-doc-check/doctrees docs/source /tmp/thefittest-doc-check/html
```

Проверьте сгенерированные страницы, ссылки и примеры перед коммитом.
Сама сборка не выполняет commit, push или публикацию сайта.
