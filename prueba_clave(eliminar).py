from services.ai_service import MockAIService

ai = MockAIService()
response = ai.evaluate_text("Evalúa la claridad de esta sección del informe institucional.")
print(response)

print(ai.calls)