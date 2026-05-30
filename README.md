# CaaS
Cutter as a Service

## Scene Dataset CLI

`cli/build_scene_dataset_v4.py` собирает `.sft`-датасет по logical scenes из видео.

### Входные данные

- `--videos-dir`: папка с видео, уже разрезанными по logical scenes (1 файл = 1 сцена)
- `--width`, `--height`: итоговый размер кадра `W x H` при чтении видео (до разбиения на center/sides)
- `--center-width`: ширина центральной области `W_C` до деления пополам. Значение должно быть чётным; `(W - W_C)` тоже должно быть чётным.

Сцены нумеруются автоматически по порядку сортировки имён файлов: `scene_logical_id = 0, 1, 2, ...`.

### Выходной формат `.sft`

Каждая часть датасета (`scene_dataset_part_00001.sft`, ...) содержит только текущий и предыдущий кадр. Дополнительные карты (`opti_map_*`, `depth_map`, `camera_move_vector`) в v4 не пишутся.

Обозначения:

- `W_center_half = W_C / 2`
- `W_side = (W - W_C) / 2`
- `N` — число samples после 2x augmentation: каждый выбранный кадр даёт левую половину и зеркальную правую половину.

Raw `.sft` из `build_scene_dataset_v4.py` содержит RGB-тензоры `float32` в диапазоне `[0..1]`:

- `source_image_center`: `[N, 3, H, W_center_half]` — левая половина центра кадра `i` или зеркальная правая половина центра
- `source_image_sides`: `[N, 3, H, W_side]` — левый бок кадра `i` или зеркальный правый бок; это target для обучения
- `previous_frame_center`: `[N, 3, H, W_center_half]` — та же половина центра кадра `i-1` (нули для первого кадра сцены)
- `previous_frame_sides`: `[N, 3, H, W_side]` — тот же бок кадра `i-1` (нули для первого кадра сцены)
- `scene_logical_id`: `[N]` (`int32`) — id логической сцены в датасете
- `frame_id`: `[N]` (`int32`) — порядковый номер кадра внутри сцены

Разбиение `source_image_*` и `previous_frame_*` выполняется одинаково. Левая половина сохраняется как есть: `left side + left half(center)`. Правая половина перед сохранением отражается по вертикальной оси: `right side + right half(center)` попадают в те же ключи, что и левые части. Благодаря этому модель учится предсказывать только одну половину кадра в единой ориентации.

После кодирования encoder/decoder API те же ключи используются как latent `.sft`, но с `C = LATENT_CHANNELS` (обычно `16`) и latent-размерами `CONDITION_*`/`QUERY_*`. Например, для side-латента шириной `8` target имеет форму `[N, 16, H_latent, 8]`.

`SFTSceneDataset` (`app/src/utils/sft_reader.py`) при чтении отдаёт поля текущего и предыдущего кадра.

## DiT diffusion training (боковые области)

Проект переходит с детерминированного UNet (`cli/train_unet_sft.py`) на **диффузионный DiT**
для генерации боковых полос текущего кадра.

### DiT v4: ключи scene `.sft`

Для `dit_v4` используется encoded latent `.sft` с теми же ключами:

- condition: `source_image_center`, форма `[N, 16, CONDITION_HEIGHT, CONDITION_WIDTH]`
- target: `source_image_sides`, форма `[N, 16, QUERY_HEIGHT, QUERY_WIDTH]`
- previous side condition: `previous_frame_sides`, форма `[N, 16, QUERY_HEIGHT, QUERY_WIDTH]`
- model input на diffusion-шаге: `noisy_target`, тот же shape, что target; шум добавляется в train loop

`dit_v4` дополнительно маскирует `previous_frame_sides` перед cross-attention: по умолчанию зануляются колонки `1,3,5,7` от правого края side-латента (`DIT_V4_SIDE_MASK_COLUMNS_FROM_RIGHT=1,3,5,7`).

Переопределение ключей: `CONDITION_KEY`, `TARGET_KEY`; размеры DiT: `CONDITION_*`, `QUERY_*`.

### Обучение

```bash
python -m cli.train_dit_v4 \
  --architecture dit_v4 \
  --dataset-dir D:/data/encoded_scene_dataset \
  --run-name dit-v4-half-frame
```

Основные параметры diffusion: `PREDICTION_TYPE` (`epsilon`, `v`, `epsilon_v_hybrid`),
`NUM_TRAIN_TIMESTEPS`, `MIN_SNR_GAMMA`, `USE_EMA`, `DETAIL_LOSS_WEIGHT`.

Логирование: MLflow experiment `DiT`, registry model `DiTModel` (см. `app/config.py`).

### Регистрация локальных весов в MLflow Registry

Если веса и конфиг уже лежат локально (без запуска train-скрипта), используйте:

```bash
python -m cli.register_dit_model_local \
  --weights-path D:/models/dit/final_model.pt \
  --config-path D:/models/dit/train_config.json \
  --registered-model-name DiTModel
```

Скрипт поднимает архитектуру из конфига, загружает `state_dict` из локального файла
и регистрирует новую версию в MLflow Model Registry.

## Training Preview (val) — UNet (legacy)

`cli/train_unet_sft.py` сохраняет превью из `N = TRAIN_PREVIEW_SAMPLES` кадров val-выборки в формате:

- `target[latent] | predict[latent]`
- `target[rgb] | predict[rgb]`
- `source_sides[rgb] - predict[rgb] | source_image_center[rgb] + (source_sides[rgb] - predict[rgb])` (сайд-полосы добавляются по бокам центра)

Количество сцен для preview регулируется через `TRAIN_PREVIEW_SCENES`,
количество кадров на каждую сцену — через `TRAIN_PREVIEW_SAMPLES`.

RGB-строки preview декодируются только через внешний API:

- `DECODER_API_ENABLED=true`
- `DECODER_API_BASE_URL=http://194.67.116.178:8001`
- `DECODER_API_TIMEOUT_SEC=120`
- `DECODER_API_CHECK_READINESS=true`

### Правила обработки v4

- Если предыдущего кадра нет, `previous_frame_center` и `previous_frame_sides` заполняются нулями.
- Разбиение на части (`--part-size-frames`) идет по числу samples после 2x augmentation, но сцена не режется посередине:
  часть закрывается только после завершения текущей логической сцены.
- Нормализация/стандартизация вынесены в явные функции:
  - RGB кадры: `float32` в диапазон `[0..1]`

### Пример запуска

```bash
python -m cli.build_scene_dataset_v4 \
  --videos-dir D:/data/logical_scenes \
  --output-dir D:/data/scene_dataset_v4 \
  --width 560 \
  --height 240 \
  --center-width 400 \
  --part-size-frames 20000 \
  --samples-per-video 2
```

## Scene Analytics CLI

`cli/build_scene_analytics.py` строит JSON-аналитику по каждому видео-сцене и каждому кадру.

### Назначение

Скрипт предназначен для последующего отбора кадров в максимально разнообразный датасет:

- отделять статичные кадры от динамичных (`diff` + `optiflow`),
- отделять реальное движение от зашумленности (`noise_ratio`/`structure_ratio`),
- учитывать наличие близких объектов на краях (`depth`/`optiflow` edge-метрики),
- анализировать полный трек движения камеры по сцене (`camera_move_track`).

### Выходной JSON формат

Верхний уровень — словарь вида:

```json
{
  "video_name_or_relative_path.mp4": {
    "param_name": [ ... ],
    "param_name_2": [ ... ]
  }
}
```

Все параметры внутри видео имеют одинаковую длину `N` (число кадров сцены), индекс в массиве = индекс кадра внутри этой сцены.

Ключевые поля:

- `camera_move_vector`: `[N, 2]` — покадровый вектор движения камеры `(dx, dy)` (depth-weighted median flow, знак инвертирован).
- `camera_move_track`: `[N, 2]` — накопленный (интегральный) трек движения камеры от начала сцены.
- `brightness_mean`, `brightness_median`: средняя и медианная яркость исходного кадра (градации серого, диапазон `[0..255]`).
- `depth_mean`, `depth_min`, `depth_max`: mean/min/max по depth карте (`MiDaS`, нормализована в `[0..1]`).
- `depth_edge_mean_left/right`, `depth_edge_p95_left/right`: значения depth на боковых краях карты.
- `diff_*`: статистики карты `absdiff(frame_i, frame_{i+1})`:
  - интенсивность изменений: `mean/median/std/p95/max`,
  - активность у краев: `edge_mean_*`, `edge_max_*`,
  - структура vs шум: `diff_noise_ratio`, `diff_structure_ratio`.
- `optiflow_*`: статистики optical flow (Farneback) между `frame_i` и `frame_{i+1}`:
  - интенсивность движения: `mean_magnitude/median_magnitude/std_magnitude/p95_magnitude/max_magnitude`,
  - среднее направление: `mean_dx`, `mean_dy`,
  - согласованность движения: `coherence` (высокое значение = более направленное/реальное движение),
  - активность по краям: `edge_mean_*`, `edge_max_*`,
  - структура vs шум: `optiflow_noise_ratio`, `optiflow_structure_ratio`.

### Пример запуска

```bash
python -m cli.build_scene_analytics \
  --videos-dir D:/data/logical_scenes \
  --output-path D:/data/scene_analytics.json \
  --max-videos 200 \
  --width 560 \
  --height 240 \
  --edge-ratio 0.1 \
  --depth-model MiDaS_small \
  --depth-device auto
```

## Scene Analytics Classifier CLI

`cli/classify_scene_analytics.py` читает готовый `scene_analytics.json` и присваивает каждой сцене классы.

Классы покрывают:

- активное / почти статичное движение камеры,
- передний / задний план (по depth),
- динамику depth по кадрам,
- яркие / тёмные сцены,
- профили `diff` (почти ноль / шум / структурные изменения),
- движение у краев кадра,
- активность пиксельного движения.

Дополнительно добавлены классы для разнообразия:

- `motion_type`: согласованное глобальное движение vs локально-хаотичное,
- `depth_contrast`: высокий контраст глубины vs плоская глубина,
- `lighting_dynamics`: динамичное vs стабильное освещение,
- `camera_drift_direction`: направление суммарного дрейфа камеры.

Пороговые значения считаются автоматически по квантилям всего набора сцен из входного JSON.

### Пример запуска

```bash
python -m cli.classify_scene_analytics \
  --analytics-json D:/data/scene_analytics.json \
  --output-path D:/data/scene_analytics_classes.json \
  --viz-dir D:/data/scene_analytics_viz
```

Если передан `--viz-dir`, скрипт дополнительно сохраняет:

- bar charts по распределению каждого семейства классов,
- scatter `camera_speed_mean` vs `flow_mean_magnitude_global` (цвет = brightness),
- scatter `depth_mean_global` vs `depth_mean_std` (цвет = diff_noise),
- гистограмму общей яркости сцен.
