"""Application-specific exceptions exposed through the API error handler."""


class AppError(Exception):
    """Base class for all app-raised exceptions."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UnsupportedLanguageError(AppError):
    def __init__(self, language: str):
        super().__init__(f"Unsupported language detected: {language}", status_code=422)
        self.language = language


class RetrievalError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=502)


class LLMGenerationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=502)
