# Configuración de seguridad para la clave de OpenAI

Este documento describe cómo preparar el repositorio para proteger la clave de
OpenAI empleada por los servicios de generación. El flujo oficial centraliza el
acceso en `utils.secret_manager.get_openai_api_key()` para garantizar el
principio de acceso controlado.

## Pasos de configuración

1. **Crear la carpeta de secretos.** Si aún no existe, genera `secrets/` en la
   raíz del repositorio; todos los artefactos sensibles viven allí.
2. **Decidir la fuente primaria de la clave.**
   - Para desarrollo rápido puedes exportar `OPENAI_API_KEY` en tu entorno.
   - Como alternativa temporal, guarda la clave en `secrets/openai_api_key.txt`.
3. **Cifrar la clave para entornos compartidos.** Ejecuta el asistente
   interactivo y sigue las instrucciones para capturar la passphrase:

   ```bash
   python -m utils.seal_key
   ```

   El asistente puede reutilizar `secrets/openai_api_key.txt`, solicitar los
   datos por consola o ejecutarse en modo no interactivo (`--non-interactive`)
   leyendo la passphrase desde una variable de entorno.
4. **Verificar la resolución.** Antes de desplegar, valida que la clave pueda
   recuperarse ejecutando una consola de Python e importando
   `utils.secret_manager.get_openai_api_key()`.

## Variables de entorno relevantes

- `OPENAI_API_KEY`: Fuente directa de la clave para escenarios de desarrollo o
  cuando la clave se inyecta por el gestor de secretos externo.
- `OPENAI_KEY_PASSPHRASE`: Passphrase necesaria para descifrar
  `secrets/openai_api_key.enc`. Debe contener al menos 8 caracteres; de lo
  contrario se rechazará.
- `PYTHONPATH` (opcional): si ejecutas los scripts desde fuera de la raíz del
  proyecto, añade el repositorio al `PYTHONPATH` para que los módulos `utils`
  sean importables.

## Flujo de cifrado y descifrado

1. `python -m utils.seal_key` delega en `utils.secret_manager.seal_key()` para
   crear `secrets/openai_api_key.enc`.
2. El módulo genera un `salt` aleatorio de 16 bytes y deriva una clave Fernet
   mediante PBKDF2-HMAC-SHA256 con 390 000 iteraciones y longitud de 32 bytes.
3. La clave derivada cifra la API key en memoria y produce un `token` con
   formato Fernet.
4. Ambos valores (`salt` codificado en base64 y `token`) se almacenan en un JSON
   dentro de `secrets/openai_api_key.enc`.
5. Durante la ejecución `get_openai_api_key()` intenta, en orden, las fuentes
   permitidas: `OPENAI_API_KEY`, `secrets/openai_api_key.txt` y el archivo
   cifrado (requiriendo `OPENAI_KEY_PASSPHRASE`).
6. Si ninguna fuente entrega la clave, el módulo lanza
   `utils.secret_manager.SecretManagerError` para evitar estados silenciosos.

Mantén la passphrase fuera del repositorio (por ejemplo, variables protegidas en
CI/CD) y rota la clave cifrada cada vez que cambie la credencial o la persona
responsable.