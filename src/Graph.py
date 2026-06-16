import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [91.89, 91.89, 91.89, 91.89]

plt.figure(figsize=(8,5))
bars = plt.barh(metrics, values)

plt.xlabel('Performance (%)')
plt.title('Detection Model Performance')
plt.xlim(0,100)

for bar, value in zip(bars, values):
    plt.text(
        value + 0.5,
        bar.get_y() + bar.get_height()/2,
        f'{value:.2f}%',
        va='center'
    )

plt.tight_layout()
plt.savefig('detection_performance_horizontal.png', dpi=300)
plt.show()