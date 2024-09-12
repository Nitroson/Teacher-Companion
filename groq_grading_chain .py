import requests
from typing import Dict, Any, Optional
import logging
from pydantic import BaseModel, Field

class GradingInput(BaseModel):
    question: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    rubric: str = Field(..., min_length=1)

class GradingConfig(BaseModel):
    model: str = "mixtral-8x7b-32768"
    temperature: float = Field(0.7, ge=0, le=1)
    max_tokens: int = Field(1000, gt=0)

class GroqAPIError(Exception):
    """Custom exception for Groq API errors."""
    pass

class GroqGradingChain:
    def __init__(self, api_key: str, config: Optional[GradingConfig] = None):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/v1"
        self.config = config or GradingConfig()
        self.logger = logging.getLogger(__name__)

    async def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        try:
            validated_inputs = GradingInput(**inputs)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": self._construct_prompt(validated_inputs)}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers) as response:
                    if response.status != 200:
                        raise GroqAPIError(f"API request failed with status {response.status}: {await response.text()}")
                    return await self._process_response(await response.json())
        except Exception as e:
            self.logger.error(f"Error in grading process: {str(e)}")
            raise

    def _construct_prompt(self, inputs: GradingInput) -> str:
        criteria = self._parse_rubric(inputs.rubric)
        return f"""
        Grade the following assignment based on the provided rubric and the question. Provide detailed feedback with marks for each section:

        Question:
        {inputs.question}

        Rubric Criteria:
        {criteria}

        Assignment Text:
        {inputs.text}

        Provide the marks and feedback for each criterion. Use the format 'Criterion: Mark' for each graded item, followed by detailed feedback.
        """

    @staticmethod
    def _parse_rubric(rubric: str) -> str:
        criteria = rubric.strip().split('\n')
        return "\n".join([f"- {criterion}" for criterion in criteria])

    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        try:
            completion_text = response["choices"][0]["message"]["content"]
            feedback_lines = completion_text.split("\n")
            grades = {}
            feedback = []
            current_criterion = None

            for line in feedback_lines:
                line = line.strip()
                if ':' in line and not current_criterion:
                    current_criterion, mark = line.split(':', 1)
                    grades[current_criterion.strip()] = mark.strip()
                elif current_criterion:
                    if ':' in line:
                        feedback.append(f"{current_criterion}: {line}")
                        current_criterion = None
                    else:
                        feedback.append(line)
                else:
                    feedback.append(line)

            return {
                "grades": grades,
                "feedback": "\n".join(feedback)
            }
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error processing API response: {str(e)}")
            raise GroqAPIError("Failed to process API response")

    @classmethod
    def from_env(cls, env_var: str = "GROQ_API_KEY", config: Optional[GradingConfig] = None):
        """
        Create a GroqGradingChain instance using an API key from an environment variable.
        """
        import os
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {env_var}")
        return cls(api_key, config)