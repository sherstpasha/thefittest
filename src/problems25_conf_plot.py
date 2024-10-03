import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из файла
file_path = "ga_cec2005.csv"
data = pd.read_csv(file_path)

# Определение порядка функций
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

# Группировка по каждому отдельному параметру и функциям, подсчёт среднего значения надёжности
reliability_table = data.groupby(['Function', 'Selection', 'Crossover', 'Mutation'])['find_solution'].mean().reset_index()

# Создание упрощённых меток групп операторов: prop | unif2
reliability_table['Group'] = (
    reliability_table['Selection'].map({
        'proportional': 'prop',
        'rank': 'rank',
        'tournament_3': 'tour3',
        'tournament_5': 'tour5',
        'tournament_7': 'tour7'
    }) + ' | ' + reliability_table['Crossover'].map({
        'empty': 'empty',
        'one_point': 'one',
        'two_point': 'two',
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

# Фильтрация по каждому типу мутации
mutation_types = ['average', 'strong', 'weak']

# Настройка фигуры с тремя подграфиками, используя `constrained_layout` для оптимизации пространства
fig, axes = plt.subplots(1, 3, figsize=(30, 20), gridspec_kw={'width_ratios': [1, 1, 1]}, constrained_layout=True)

# Построение каждой тепловой карты на своём подграфике
for i, mutation in enumerate(mutation_types):
    # Фильтруем строки по текущему типу мутации
    filtered_table = reliability_table[reliability_table['Mutation'] == mutation]
    
    # Создание сводной таблицы: строки — группы операторов, столбцы — функции, значения — надёжность
    pivot_table = filtered_table.pivot(index='Group', columns='Function', values='find_solution')
    
    # Заполнение пропущенных значений (если какие-то комбинации не использовались)
    pivot_table = pivot_table.fillna(0)

    # Добавление столбца среднего значения по всем задачам и название символом X̅
    pivot_table['X̅'] = pivot_table.mean(axis=1)

    # Построение тепловой карты на соответствующем подграфике
    sns.heatmap(pivot_table, cmap='YlGnBu', cbar=i == 2,  # Показать цветовой бар только на последнем графике
                annot=False, xticklabels=True, yticklabels=(i == 0), ax=axes[i])
    axes[i].set_title(f'{mutation.capitalize()} Mutation')
    axes[i].set_xlabel('Функции (порядковый номер)')
    
    # Убираем подпись оси Y на всех графиках, кроме первого
    if i == 0:
        axes[i].set_ylabel('Группы операторов')
    else:
        axes[i].set_ylabel('')  # Убираем подпись 'Group' на остальных

# Устанавливаем общий заголовок для всей фигуры
plt.suptitle('Тепловая карта надёжности операторов по типам мутаций', fontsize=24)

# Показ графика
plt.show()

# Сохранение каждой таблицы отдельно для последующего анализа
for mutation in mutation_types:
    filtered_table = reliability_table[reliability_table['Mutation'] == mutation]
    pivot_table = filtered_table.pivot(index='Group', columns='Function', values='find_solution')
    pivot_table = pivot_table.fillna(0)
    pivot_table['X̅'] = pivot_table.mean(axis=1)
    pivot_table.to_csv(f"problem25_reliability_table_{mutation}.csv")
