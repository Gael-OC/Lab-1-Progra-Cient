import os
from sklearn.datasets import fetch_20newsgroups

# Categorías seleccionadas
categories = [
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "sci.electronics",
    "comp.sys.mac.hardware"
]

# Crear carpeta raíz donde guardaremos todo
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

dataset = fetch_20newsgroups(
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=()  # Puede probar con ("headers", "footers", "quotes") para quitar algunos fragmentos de los textos
)

# Guardar cada documento como txt en su carpeta
for i, text in enumerate(dataset.data):
    label = dataset.target_names[dataset.target[i]]
    folder = os.path.join(output_dir, label)
    os.makedirs(folder, exist_ok=True)
    
    filename = os.path.join(folder, f"doc_{i}.txt")
    with open(filename, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)