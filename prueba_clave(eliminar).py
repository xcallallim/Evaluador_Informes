from services.ai_service import MockAIService

ai = MockAIService()
response = ai.evaluate("Texto de ejemplo institucional.")
print(response)