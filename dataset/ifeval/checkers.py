import re
import json
from typing import List, Dict, Optional, Tuple, Any

class BaseChecker:
    def check(self, response: str, constraints: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Checks if the response satisfies the constraints.
        Returns: (passed, error_message)
        """
        raise NotImplementedError

class TabooChecker(BaseChecker):
    def check(self, response: str, constraints: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Checks if the response contains any forbidden words.
        Constraints schema:
        {
            "forbidden_words": ["word1", "word2"]
        }
        """
        forbidden_words = constraints.get("forbidden_words", [])
        if not forbidden_words:
            return True, None

        # Normalize response to lowercase for case-insensitive matching
        response_lower = response.lower()
        
        violations = []
        for word in forbidden_words:
            # Use regex for word boundary matching to avoid partial matches (e.g. banning "cat" shouldn't ban "category")
            # Escape the word to handle special characters
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            if re.search(pattern, response_lower):
                violations.append(word)
        
        if violations:
            return False, f"Found forbidden words: {', '.join(violations)}"
        
        return True, None

class JsonChecker(BaseChecker):
    def check(self, response: str, constraints: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Checks if the response is valid JSON and optionally contains required keys.
        Constraints schema:
        {
            "required_keys": ["key1", "key2"]  # Optional
        }
        """
        # 1. Extract JSON candidate
        # Try to find content within ```json ... ``` or just the first {...}
        json_str = response.strip()
        
        # Regex to find code blocks
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to find the first '{' and last '}'
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
            else:
                # If no braces found, it's definitely not a JSON object
                return False, "No JSON object found (missing braces)"

        # 2. Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON syntax: {str(e)}"
        
        if not isinstance(data, dict):
             return False, "JSON must be an object (dict), not a list or primitive"

        # 3. Check Keys
        required_keys = constraints.get("required_keys", [])
        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            return False, f"Missing required keys: {', '.join(missing_keys)}"
            
        return True, None

def get_checker(task_type: str) -> BaseChecker:
    if task_type == "forbidden_words":
        return TabooChecker()
    elif task_type == "json_format":
        return JsonChecker()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
