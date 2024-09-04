import requests

class GroqGradingChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.ai/v1"  # Hypothetical API URL, replace with actual URL if known

    def __call__(self, inputs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "groq-v1",  # Hypothetical model name
            "prompt": self.construct_prompt(inputs)
        }
        response = requests.post(f"{self.base_url}/completions", json=payload, headers=headers)
        if response.status_code == 200:
            return self._process_response(response.json())
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    def construct_prompt(self, inputs):
        question = inputs['question']
        text = inputs['text']
        rubric = inputs['rubric']
        criteria = self.parse_rubric(rubric)
        prompt = f"""
        \n\nHuman: Grade the following assignment based on the provided rubric and the question. Provide detailed feedback with marks for each section:

        **Question:**
        {question}

        **Rubric Criteria:**
        {criteria}

        **Assignment Text:**
        {text}

        Provide the marks and feedback for each criterion.
        \n\nAssistant:
        """
        return prompt

    def parse_rubric(self, rubric):
        criteria = rubric.strip().split('\n')
        formatted_criteria = "\n".join([f"- {criterion}" for criterion in criteria])
        return formatted_criteria

    def _process_response(self, response):
        completion_text = response.get("choices", [{}])[0].get("text", "")
        feedback_lines = completion_text.split("\n")
        grades = {}
        feedback = []
        for line in feedback_lines:
            if ':' in line:
                section, mark = line.split(':', 1)
                grades[section.strip()] = mark.strip()
            else:
                feedback.append(line.strip())

        return {
            "grades": grades,
            "feedback": "\n".join(feedback)
        }
