# data/models/evaluation.py

"""Este archivo se enfoca en la estructura del resultado de la evaluación. Al igual que con document.py, 
su objetivo es asegurar que el resultado de la evaluación siempre tenga el mismo formato, lo que 
facilita el trabajo de otros módulos (como el que calcula el índice o el que lo muestra en la interfaz)."""

from typing import Dict, List, Optional
from datetime import datetime

# Una clase para representar el resultado de la evaluación de un solo criterio.
class CriteriaResult:
    def __init__(
        self,
        criteria_name: str,
        score: int,                  # El puntaje asignado (ej. de 1 a 5)
        justification: Optional[str] = None, # Explicación de por qué se asignó ese puntaje
        relevant_text: Optional[str] = None  # Texto del documento que justificó el puntaje
    ):
        self.criteria_name = criteria_name
        self.score = score
        self.justification = justification
        self.relevant_text = relevant_text

# La clase principal que agrupa todos los resultados de la evaluación.
class Evaluation:
    def __init__(
        self,
        document_id: str,
        evaluation_date: datetime = datetime.now(),
        final_index: Optional[float] = None, # El índice calculado con todos los puntajes
        criteria_results: List[CriteriaResult] = None,
    ):
        self.document_id = document_id
        self.evaluation_date = evaluation_date
        self.final_index = final_index
        self.criteria_results = criteria_results or []