"""
Confusion Matrix Calculator.

This module provides functionality to compute confusion matrices and derived metrics
for classification tasks. Supports both binary and multi-class classification.
"""

import numpy as np
from typing import Dict, Tuple, Union, Optional
from dataclasses import dataclass


@dataclass
class ConfusionMatrixResult:
    """
    Resultado de cálculo de matriz de confusión con métricas derivadas.

    Este dataclass encapsula la matriz de confusión y todas las métricas
    derivadas como precision, recall, F1-score, etc.
    """
    matrix: np.ndarray
    """Matriz de confusión con valores absolutos (n_classes, n_classes)"""

    matrix_normalized_row: np.ndarray
    """Matriz normalizada por fila (recall por clase)"""

    matrix_normalized_col: np.ndarray
    """Matriz normalizada por columna (precision por clase)"""

    precision: Dict[str, float]
    """Precision por clase: TP / (TP + FP)"""

    recall: Dict[str, float]
    """Recall por clase: TP / (TP + FN)"""

    specificity: Dict[str, float]
    """Especificidad por clase: TN / (TN + FP)"""

    f1_score: Dict[str, float]
    """F1-score por clase: 2 * (precision * recall) / (precision + recall)"""

    support: Dict[str, int]
    """Número de muestras reales por clase"""

    accuracy: float
    """Accuracy global: (TP + TN) / total"""

    macro_avg: Dict[str, float]
    """Promedio macro de precision, recall, f1"""

    weighted_avg: Dict[str, float]
    """Promedio ponderado por support de precision, recall, f1"""

    n_classes: int
    """Número de clases"""


class ConfusionMatrixCalculator:
    """
    Calculadora de matriz de confusión y métricas derivadas.
    
    Esta clase calcula la matriz de confusión para problemas de clasificación
    binarios y multi-clase, junto con métricas como precision, recall, F1-score,
    accuracy, etc.
    """
    
    def __init__(self):
        """Inicializar la calculadora."""
        pass
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[np.ndarray] = None,
        discretize: bool = True
    ) -> ConfusionMatrixResult:
        """
        Calcular matriz de confusión y métricas derivadas.
        
        Args:
            y_true: Valores verdaderos de forma (n_samples,) o (n_samples, 1)
            y_pred: Valores predichos de forma (n_samples,) o (n_samples, 1)
            labels: Etiquetas de clases opcionales. Si es None, se infieren de y_true
            discretize: Si discretizar predicciones continuas a clases cercanas
            
        Returns:
            ConfusionMatrixResult con matriz y métricas
            
        Raises:
            ValueError: Si y_true y y_pred tienen formas incompatibles
        """
        # Validar formas
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true y y_pred deben tener la misma forma. "
                f"y_true: {y_true.shape}, y_pred: {y_pred.shape}"
            )
        
        # Aplanar si es necesario
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Discretizar predicciones si es necesario
        if discretize:
            y_pred = self._discretize_predictions(y_pred, y_true)
        
        # Inferir etiquetas si no se proporcionan
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        
        n_classes = len(labels)
        
        # Calcular matriz de confusión
        matrix = self._compute_matrix(y_true, y_pred, labels)
        
        # Calcular matrices normalizadas
        matrix_normalized_row = self._normalize_by_row(matrix)
        matrix_normalized_col = self._normalize_by_column(matrix)
        
        # Calcular métricas por clase
        precision, recall, specificity, f1, support = self._compute_class_metrics(matrix, labels)

        # Calcular accuracy global
        accuracy = np.trace(matrix) / np.sum(matrix)

        # Calcular promedios
        macro_avg = self._compute_macro_avg(precision, recall, specificity, f1)
        weighted_avg = self._compute_weighted_avg(precision, recall, specificity, f1, support)

        return ConfusionMatrixResult(
            matrix=matrix,
            matrix_normalized_row=matrix_normalized_row,
            matrix_normalized_col=matrix_normalized_col,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1_score=f1,
            support=support,
            accuracy=accuracy,
            macro_avg=macro_avg,
            weighted_avg=weighted_avg,
            n_classes=n_classes
        )
    
    def _compute_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Calcular matriz de confusión con valores absolutos.
        
        Args:
            y_true: Valores verdaderos aplanados
            y_pred: Valores predichos aplanados
            labels: Etiquetas de clases
            
        Returns:
            Matriz de confusión (n_classes, n_classes)
        """
        n_classes = len(labels)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # Crear mapeo de etiqueta a índice
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Llenar matriz
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            matrix[true_idx, pred_idx] += 1
        
        return matrix
    
    def _discretize_predictions(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Discretizar predicciones continuas a las clases más cercanas en y_true.
        
        Esto es útil cuando las redes neuronales devuelven valores continuos
        en lugar de etiquetas de clase discretas.
        
        Args:
            y_pred: Predicciones (pueden ser continuas)
            y_true: Valores verdaderos (etiquetas discretas)
            
        Returns:
            Predicciones discretizadas a las clases más cercanas
        """
        unique_classes = np.unique(y_true)
        
        if len(unique_classes) == 2:
            # Clasificación binaria: usar umbral 0.5
            threshold = 0.5
            y_pred_discrete = np.where(y_pred >= threshold, unique_classes[1], unique_classes[0])
        else:
            # Multi-clase: asignar a la clase más cercana
            y_pred_discrete = np.zeros_like(y_pred)
            for pred in y_pred:
                # Encontrar la clase más cercana
                closest_class = unique_classes[np.argmin(np.abs(unique_classes - pred))]
                y_pred_discrete[np.where(y_pred == pred)[0][0]] = closest_class
        
        return y_pred_discrete
    
    def _normalize_by_row(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizar matriz por fila (recall por clase).
        
        Args:
            matrix: Matriz de confusión
            
        Returns:
            Matriz normalizada por fila
        """
        normalized = matrix.astype(float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        
        # Evitar división por cero
        row_sums[row_sums == 0] = 1
        normalized = normalized / row_sums
        
        return normalized
    
    def _normalize_by_column(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizar matriz por columna (precision por clase).
        
        Args:
            matrix: Matriz de confusión
            
        Returns:
            Matriz normalizada por columna
        """
        normalized = matrix.astype(float)
        col_sums = matrix.sum(axis=0, keepdims=True)
        
        # Evitar división por cero
        col_sums[col_sums == 0] = 1
        normalized = normalized / col_sums
        
        return normalized
    
    def _compute_class_metrics(
        self,
        matrix: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Calcular métricas por clase: precision, recall, specificity, F1-score, support.

        Args:
            matrix: Matriz de confusión
            labels: Etiquetas de clases

        Returns:
            Tupla con (precision, recall, specificity, f1_score, support) como diccionarios
        """
        precision = {}
        recall = {}
        specificity = {}
        f1_score = {}
        support = {}

        for idx, label in enumerate(labels):
            tp = matrix[idx, idx]  # True Positives
            fp = matrix[:, idx].sum() - tp  # False Positives
            fn = matrix[idx, :].sum() - tp  # False Negatives
            tn = matrix.sum() - tp - fp - fn  # True Negatives

            support[str(label)] = int(matrix[idx, :].sum())

            # Precision: TP / (TP + FP)
            if tp + fp > 0:
                precision[str(label)] = tp / (tp + fp)
            else:
                precision[str(label)] = 0.0

            # Recall: TP / (TP + FN)
            if tp + fn > 0:
                recall[str(label)] = tp / (tp + fn)
            else:
                recall[str(label)] = 0.0

            # Specificity: TN / (TN + FP)
            if tn + fp > 0:
                specificity[str(label)] = tn / (tn + fp)
            else:
                specificity[str(label)] = 0.0

            # F1-score: 2 * (precision * recall) / (precision + recall)
            if precision[str(label)] + recall[str(label)] > 0:
                f1_score[str(label)] = (
                    2 * precision[str(label)] * recall[str(label)] /
                    (precision[str(label)] + recall[str(label)])
                )
            else:
                f1_score[str(label)] = 0.0

        return precision, recall, specificity, f1_score, support
    
    def _compute_macro_avg(
        self,
        precision: Dict[str, float],
        recall: Dict[str, float],
        specificity: Dict[str, float],
        f1: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcular promedio macro de métricas.

        Args:
            precision: Diccionario de precision por clase
            recall: Diccionario de recall por clase
            specificity: Diccionario de specificity por clase
            f1: Diccionario de F1-score por clase

        Returns:
            Diccionario con promedios macro
        """
        return {
            'precision': np.mean(list(precision.values())),
            'recall': np.mean(list(recall.values())),
            'specificity': np.mean(list(specificity.values())),
            'f1-score': np.mean(list(f1.values()))
        }
    
    def _compute_weighted_avg(
        self,
        precision: Dict[str, float],
        recall: Dict[str, float],
        specificity: Dict[str, float],
        f1: Dict[str, float],
        support: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calcular promedio ponderado por support de métricas.

        Args:
            precision: Diccionario de precision por clase
            recall: Diccionario de recall por clase
            specificity: Diccionario de specificity por clase
            f1: Diccionario de F1-score por clase
            support: Diccionario de support por clase

        Returns:
            Diccionario con promedios ponderados
        """
        total_support = sum(support.values())

        if total_support == 0:
            return {'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1-score': 0.0}

        weighted_precision = sum(
            precision[label] * support[label] for label in precision
        ) / total_support

        weighted_recall = sum(
            recall[label] * support[label] for label in recall
        ) / total_support

        weighted_specificity = sum(
            specificity[label] * support[label] for label in specificity
        ) / total_support

        weighted_f1 = sum(
            f1[label] * support[label] for label in f1
        ) / total_support

        return {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'specificity': weighted_specificity,
            'f1-score': weighted_f1
        }
