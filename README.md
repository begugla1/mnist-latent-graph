# Восстановление графа MNIST

Версия для Git: код, два отчёта PDF, LaTeX-исходники и векторные рисунки.
Датасет, архивы, сырые результаты, веса моделей и кэши не включены.

## С чего начать

- `algorithm_article_portrait_ru.pdf` — алгоритм, формулы и шаги поиска.
- `experiment_article_portrait_ru.pdf` — протокол, исторические измерения и ограничения.
- `src/run_mnist.py` — основной запуск эксперимента.
- `src/latent_graph.py` — Lasso, Ridge-энергия, pruning и отжиг.
- `src/mnist_components.py` — загрузка MNIST, метрики и замороженные GNN.
- `original_sources/` — исходники проекта для сверки; старые пути и генераторы
  отчётов сохранены как историческая справка, не основной запуск.

## Запуск

Python 3.12. Из этой папки:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/check_package.py
.\.venv\Scripts\python.exe src/run_mnist.py --zip C:\datasets\mnist.zip --smoke --output smoke_run
.\.venv\Scripts\python.exe src/run_mnist.py --zip C:\datasets\mnist.zip
```

На Linux/macOS путь интерпретатора — `.venv/bin/python`.
MNIST предоставить отдельно. ZIP должен содержать четыре распакованных IDX-файла:
`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`,
`t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`.
Допускаются подпапки внутри ZIP. Файлы `.gz` вместо IDX загрузчик не читает.

Полный запуск выполняет девять восстановлений и шесть обучений GNN.
GNN обучается только на согласованной решётке; далее графы сравниваются
с фиксированными весами, без дообучения. Результаты записываются в `reruns/`.
`--smoke` — техническая проверка, не исследовательские результаты.
Численные проверки не требуют датасета. Для проверки архива:
`python src/check_package.py --zip C:\datasets\mnist.zip`.

## Пересборка PDF

Датасет не нужен: таблицы `.tex` и рисунки `.pdf` сохранены.
Установить Tectonic либо XeLaTeX и зависимости:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe build_reports.py --engine C:\path\to\tectonic.exe
```

Если компилятор в PATH, `--engine` можно опустить. При первом запуске Tectonic
загружает TeX-пакеты. При проблеме с переадресацией можно добавить
`--bundle https://data1b.fullyjustified.net/tlextras-2022.0r0.tar`.
В `reports/style.tex` используются Times New Roman, Arial, Consolas;
на другой ОС установить эти шрифты или заменить настройки.
Скрипт также проверяет границы страниц; служебные файлы исключены из Git.

Для пересоздания рисунков и таблиц нужны внешние исторические артефакты:

```powershell
python src/build_assets.py --zip C:\datasets\mnist.zip --results-dir C:\experiments\mnist_results
```

В указанной папке должны быть `repeated_metrics.json` и `pilot_graph.npz`
в формате исторического эксперимента. Без них доступны готовые PDF и LaTeX-сборка,
но повторная проверка исходных измерений невозможна.

## Происхождение и ограничения

Числа отчёта — сохранённые три повторных MNIST-прогона, не новые измерения.
В текущем ядре после них появились обработка неопределённых корреляций и диагностика
стадий. Побитовое повторение исторических графов не гарантируется.
Веса исторических GNN и рёбра второго/третьего seed не сохранялись.
`check_package.py` сравнивает вычислительный код с доступным снимком исходников.

Обе GNN используют `flatten + Linear`: accuracy не доказывает необходимость
message passing. Вариант GraphSAGE использует симметричную нормировку, не обычное
среднее соседей. Correlation baseline получает бюджет рёбер от recovery;
случайный baseline сохраняет число рёбер, а не распределение степеней.

Для публикации отдельного репозитория использовать эту папку.
`.gitignore` исключает данные, архивы, окружения, веса и результаты запусков.
