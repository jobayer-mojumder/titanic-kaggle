# type: ignore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("results_summary.csv")
df = df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="features", y="accuracy", hue="model", dodge=False)
plt.xticks(rotation=90)
plt.title("Model Accuracy by Feature Set")
plt.tight_layout()
plt.savefig("results_plot.png")
plt.show()
