import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['BCW','CR','BA','DDS','TN','RN']
methods = [
    'PDPGP',
    'SelfCGP',
    'SelfCSHAGP'
]
#  1 = PDPSHAGP лучше, 0 = нет различий, -1 = PDPSHAGP хуже
data = {
    'PDPGP':    [0, 0, 0, -1, 1, 1],
    'SelfCGP':  [0, 1, 0, 0,1, 1],
    'SelfCSHAGP':[0, 0, 0, 0, 0, 0]
}

fig, ax = plt.subplots(figsize=(6,2.5))
hatches = {1:'//', 0:'...', -1:'\\\\'}
colors  = {1:'green', 0:'lightgray', -1:'red'}

for i, m in enumerate(methods):
    for j, d in enumerate(datasets):
        val = data[m][j]
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             facecolor=colors[val],
                             edgecolor='black',
                             hatch=hatches[val])
        ax.add_patch(rect)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.set_xlim(-0.5, len(datasets)-0.5)
ax.set_ylim(len(methods)-0.5, -0.5)
ax.set_title('Сравнение PDPSHAGP с альтернативными методами\n(p < 0.05)')

legend = [
    mpatches.Patch(facecolor='green',    hatch='//',  label='PDPSHAGP лучше'),
    mpatches.Patch(facecolor='lightgray',hatch='...', label='Нет различий'),
    mpatches.Patch(facecolor='red',      hatch='\\\\',label='PDPSHAGP хуже')
]
ax.legend(handles=legend, bbox_to_anchor=(1.02,1), loc='upper left')

plt.tight_layout()
plt.show()
