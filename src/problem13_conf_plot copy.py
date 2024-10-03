import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из файла
file_path = "shagaconf_cec2005.csv"
data = pd.read_csv(file_path)

# Определение порядка функций (в нужной последовательности)
function_order = [
    "ShiftedRosenbrock", "ShiftedRotatedGriewank", "ShiftedExpandedGriewankRosenbrock",
    "RotatedVersionHybridCompositionFunction1", "RotatedVersionHybridCompositionFunction1Noise",
    "RotatedHybridCompositionFunction", "HybridCompositionFunction3", "HybridCompositionFunction3H",
    "NonContinuousHybridCompositionFunction3", "HybridCompositionFunction4",
    "HybridCompositionFunction4withoutbounds", "Rosenbrock", "ExpandedScaffers_F6",
    "Weierstrass", "ShiftedSphere", "ShiftedSchwefe1_2", "ShiftedSchwefe1_2WithNoise",
    "ShiftedRastrigin", "ShiftedRotatedRastrigin", "HybridCompositionFunction1", "Sphere",
    "HighConditionedElliptic", "Griewank", "Ackley", "Rastrigin"
]

# Присваивание порядковых номеров каждому названию функции
function_mapping = {name: idx + 1 for idx, name in enumerate(function_order)}

# Замена названий функций на их порядковые номера в исходных данных
data['Function'] = data['Function'].map(function_mapping)

# Создание упрощённых меток групп операторов: prop | unif2
data['Group'] = (
    data['Selection'].map({
        'proportional': 'prop',
        'rank': 'rank',
        'tournament_3': 'tour3',
        'tournament_5': 'tour5',
        'tournament_7': 'tour7'
    }) + ' | ' + data['Crossover'].map({
        'empty': 'empty',
        'one_point': 'one',
        'two_point': 'two',
        'uniform_1': 'unif1',
        'uniform_2': 'unif2',
        'uniform_7': 'unif7',
        'uniform_prop_2': 'up2',
        'uniform_prop_7': 'up7',
        'uniform_rank_2': 'ur2',
        'uniform_rank_7': 'ur7',
        'uniform_tour_3': 'ut3',
        'uniform_tour_7': 'ut7'
    })
)

# Группировка данных и устранение дубликатов с помощью среднего значения надёжности
reliability_table = data.groupby(['Function', 'Group'])['find_solution'].mean().reset_index()

# Создание сводной таблицы: строки — группы операторов, столбцы — функции, значения — надёжность
pivot_table = reliability_table.pivot(index='Group', columns='Function', values='find_solution')

# Заполнение пропущенных значений (если какие-то комбинации не использовались)
pivot_table = pivot_table.fillna(0)

# Удаление пустых строк (если есть группы, которые не использовались в данных)
pivot_table = pivot_table[(pivot_table.T != 0).any()]

# Добавление столбца среднего значения по всем задачам и название символом X̅
pivot_table['X̅'] = pivot_table.mean(axis=1)

# Построение единственной тепловой карты с оптимизированными параметрами
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_table, cmap='YlGnBu', cbar=True, annot=False, xticklabels=True, yticklabels=True)
plt.title('Тепловая карта надёжности операторов на различных задачах', fontsize=24)
plt.xlabel('Функции (порядковый номер)')
plt.ylabel('Группы операторов')
plt.xticks(rotation=0)  # Устанавливаем вертикальное отображение порядковых номеров функций
plt.yticks(rotation=0)  # Оставляем подписи групп горизонтальными

# Устанавливаем компактную компоновку для минимизации пустого пространства
plt.tight_layout()

# Сохранение и показ графика
plt.show()

# Сохранение итоговой таблицы с усреднёнными значениями надёжности
pivot_table.to_csv("shaga_conf_problems13_reliability_table_optimized.csv", index=True)
