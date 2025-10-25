from services.evaluation_service import EvaluationService

service = EvaluationService()
result = service.run(
    input_path="data/input/test_loader.txt",
    tipo_informe="institucional",
    mode="global"
)