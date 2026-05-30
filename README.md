# CaaS
Cutter as a Service

## Scene Dataset CLI

`cli/build_scene_dataset.py` собирает `.sft`-датасет по logical scenes из видео и `advanced markup`.

### Входные данные

- `--videos-dir`: папка с видео, уже разрезанными по logical scenes (1 файл = 1 сцена)
- `--width`, `--height`: итоговый размер кадра `W x H` при чтении видео (до разбиения на center/sides)
- `--center-width`: ширина центральной области `W_C` (симметричный crop по ширине; `(W - W_C)` должно быть чётным)

Сцены нумеруются автоматически по порядку сортировки имён файлов: `scene_logical_id = 0, 1, 2, ...`.

### Выходной формат `.sft`

Каждая часть датасета (`scene_dataset_part_00001.sft`, ...) содержит тензоры:

- `source_image_center`: `[N, 16, H, W_C]` — кадр `i`, центральная полоса (симметричный crop)
- `source_image_sides`: `[N, 16, H, W - W_C]` — боковые области кадра `i` (target для обучения), слева+справа (stack по ширине)
- `opti_map_1`: `[N, 16, H, W_C]` — optical flow между `i` и `i+2` (flow считается после center-crop)
- `opti_map_2`: `[N, 16, H/2, W_C/2]` — optical flow между `i` и `i+4`, затем downscale с усреднением
- `depth_map`: `[N, 16, H/2, W_C/2]` — depth map для кадра `i` (по center-crop)
- `camera_move_vector`: `[N, 2]` — вектор движения камеры `(dx, dy)`:
  `median(opti_map_2 * (1 - depth_map))`, знак инвертирован (камера vs движение пикселей)
- `previous_frame_center`: `[N, 16, H, W_C]` — центр кадра `i-1` (нули для первого кадра сцены)
- `previous_frame_sides`: `[N, 16, H, W - W_C]` — боковые области кадра `i-1` (нули, если кадра нет)
- `scene_logical_id`: `[N]` (`int32`) — id логической сцены в датасете
- `frame_id`: `[N]` (`int32`) — порядковый номер кадра внутри сцены

Разбиение `source_image_*` и `previous_frame_*` выполняется одинаково: при `left = (W - W_C) // 2` центр —
`[:, left:left+W_C]`, боковины — `concat([:, :left], [:, left+W_C:])` по ширине.

`SFTSceneDataset` (`app/src/utils/sft_reader.py`) при чтении отдаёт поля текущего и предыдущего кадра.
Старые части с единым ключом `source_frame` по-прежнему поддерживаются: center/sides вычисляются на лету.

## DiT diffusion training (боковые области)

Проект переходит с детерминированного UNet (`cli/train_unet_sft.py`) на **диффузионный DiT**
для генерации боковых полос текущего кадра.

### Baseline: ключи scene `.sft`

Используется тот же `.sft`, что собирает `build_scene_dataset` (остальные поля сохраняются для будущих экспериментов):

| Роль | Ключ | Форма (пример) |
|------|------|----------------|
| condition | `source_image_center` | `[N, 16, H, W_C]` |
| target (чистый) | `source_image_sides` | `[N, 16, H, W_sides]` |
| model input на шаге | `noisy_target` | тот же shape, что target — шум добавляется в train loop |

Переопределение ключей: `CONDITION_KEY`, `TARGET_KEY`; размеры DiT: `CONDITION_*`, `QUERY_*`.

### Обучение

```bash
python -m cli.train_dit \
  --architecture dit_v2 \
  --dataset-dir D:/data/encoded_scene_dataset \
  --run-name dit-v2-sides-baseline
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

### Правила обработки

- Если кадра для любого поля не хватает (`i+1`, `i+2`, `i+4`), соответствующее поле заполняется нулями.
- Разбиение на части (`--part-size-frames`) идет по числу кадров, но сцена не режется посередине:
  часть закрывается только после завершения текущей логической сцены.
- Нормализация/стандартизация вынесены в явные функции:
  - RGB кадры: `float32` в диапазон `[0..1]`
  - depth: `float32` `[0..1]`
  - flow: нормализация по размеру карты (`dx/W`, `dy/H`)
  - camera vector: нормализация по размеру `opti_map_2`

### Пример запуска

```bash
python -m cli.build_scene_dataset \
  --videos-dir D:/data/logical_scenes \
  --output-dir D:/data/scene_dataset \
  --width 560 \
  --height 240 \
  --center-width 400 \
  --part-size-frames 20000 \
  --num-workers 8 \
  --max-in-flight 32
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
