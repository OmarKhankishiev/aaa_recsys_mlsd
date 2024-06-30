## MLFlow & AirFlow Task

1. Сборка проекта:

```bash
make build
```

2. Запуск прогноза (лучший по предыдущему заданию):

```bash
make run
```

3. [Опционально] Можно запустить и другие модели:
```
python script.py --experiment_name <NAME> --run_name <NAME> --model_name <NAME> --optimizer_name <Name>
```

### Список возможных моделей
- best\_model (Дефолтная модель)
- baseline\_LFM
- baseline\_top\_popular
- LFM
- ALS
- optuna\_LFM\_search
- optuna\_ALS\_search

### Список возможных оптимизаторов (только для модели `LFM`)

- Adam
- SGD
- RMSprop
- AdamW
- Adagrad

