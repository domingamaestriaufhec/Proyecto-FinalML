# Proyecto Final ML - Retención de Vocabulario con Machine Learning

Aplicación de Machine Learning supervisado para predecir y optimizar la retención de vocabulario en segundas lenguas (L2) mediante la teoría de la **Curva del Olvido de Ebbinghaus** y el algoritmo de **Repetición Espaciada**.

## 🧠 Descripción del Proyecto

Este proyecto implementa un sistema de IA que predice en tiempo real cuándo un estudiante está a punto de olvidar una palabra en inglés y recomienda el intervalo óptimo de repaso personalizado.

## 📁 Estructura del Repositorio

```
├── notebook.ipynb          # Análisis ML completo (ETL, EDA, 4 modelos supervisados)
├── app/
│   ├── index.html          # Aplicación web interactiva (flashcards + dashboard)
│   ├── style.css           # Estilos premium con diseño oscuro moderno
│   └── script.js           # Motor de inferencia ML local en JavaScript
└── Material 1.docx         # Documentación del proyecto
```

## 🚀 Modelos de Machine Learning Implementados

1. **Regresión Logística** - Modelo base probabilístico e interpretable
2. **Árbol de Decisión** - Clasificador basado en reglas pedagógicas
3. **Random Forest** - Ensamble robusto (mejor rendimiento: AUC-ROC > 0.98)
4. **K-Nearest Neighbors (KNN)** - Clasificador por similitud de instancias

## 📊 Resultados Clave

| Modelo             | Accuracy | F1-Score | AUC-ROC |
|--------------------|----------|----------|---------|
| Regresión Logística | 95.7%   | 95.3%    | 0.9938  |
| Random Forest       | 94.1%   | 93.6%    | 0.9871  |
| KNN                 | 89.3%   | 87.9%    | 0.9601  |
| Árbol de Decisión   | 89.4%   | 88.3%    | 0.9553  |

## 🌐 Aplicación Web

La app incluye:
- 🎴 **Práctica Activa** - Tarjetas de vocabulario con medición de latencia en ms
- 🎛️ **Simulador ML** - Predicciones en tiempo real con sliders interactivos
- 📊 **Panel de Retención** - Dashboard con estado de cada palabra

## 📦 Tecnologías Utilizadas

- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn, ipywidgets
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML**: Clasificación Binaria Supervisada

## ✍️ Autor

**domingamaestriaufhec**  
Maestría en Inteligencia Artificial — UFHEC  
Aprendizaje Autónomo — 2026
