# Восстановление графа по данным MNIST

Эксперимент по восстановлению соседства пикселей без их координат. Пиксели во всех изображениях перемешиваются одной перестановкой. Алгоритм получает матрицу яркостей и строит разреженный граф, используя только значения признаков.

Начальное соседство выбирается через корреляции и Lasso. Затем рёбра добавляются и удаляются по критерию, который объединяет ошибку Ridge-предсказания и штраф за число рёбер.

Полученный граф сравнивается с исходной решёткой. Дополнительно GCN и GraphSAGE обучаются на правильной решётке и проверяются с разными графами **без дообучения**.

## Код и отчёты

- [src/latent_graph.py](src/latent_graph.py) — алгоритм восстановления, функция `recover_graph(X, X_val, config)`.
- [src/run_mnist.py](src/run_mnist.py) — разбиение данных, выбор параметров и запуск эксперимента.
- [src/mnist_components.py](src/mnist_components.py) — загрузка MNIST, модели и метрики.
- [Описание алгоритма](algorithm_article_portrait_ru.pdf) — шаги и формулы.
- [Результаты эксперимента](experiment_article_portrait_ru.pdf) — сравнение графов и метрики трёх запусков.

В `original_sources/` оставлены исходные версии скриптов для сверки. Для запуска используется код из `src/`.

## Запуск

Нужен Python 3.12. Команды выполняются из этой папки, желательно в отдельном виртуальном окружении:

```bash
python -m pip install -r requirements.txt
python src/run_mnist.py --zip /path/to/mnist.zip
```

Датасет в репозиторий не включён. Загрузчик ожидает ZIP с четырьмя IDX-файлами:

```text
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
```

Если файлы скачаны в формате `.gz`, их нужно распаковать перед добавлением в ZIP.

Полный запуск проверяет три перестановки и три значения штрафа за рёбра. Метрики и графы сохраняются в `reruns/`; другую папку можно указать через `--output`.

Для быстрой проверки на небольшой выборке:

```bash
python src/run_mnist.py --zip /path/to/mnist.zip --smoke --output smoke_run
```

Проверки формулы Ridge, структуры графа и соответствия исходникам запускаются отдельно, без MNIST:

```bash
python src/check_package.py
```

## Сборка отчётов

LaTeX-исходники находятся в `reports/`, рисунки — в `figures/`. Для пересборки нужен Tectonic или XeLaTeX:

```bash
python -m pip install -r requirements-report.txt
python build_reports.py
```

Если компилятор не в PATH, его путь задаётся через `--engine`. Шрифты указаны в `reports/style.tex`: Times New Roman, Arial и Consolas.

PDF собираются без датасета, из сохранённых таблиц и рисунков. Чтобы пересоздать сами таблицы и рисунки, нужны исходные результаты `repeated_metrics.json`, `pilot_graph.npz` и MNIST:

```bash
python src/build_assets.py --zip /path/to/mnist.zip --results-dir /path/to/results
```

Данные и результаты запусков исключены из Git. Числа в PDF относятся к ранее проведённым экспериментам.
