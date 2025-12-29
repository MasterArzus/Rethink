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

        # Remove <|end_of_sentence|> to avoid false positives if forbidden words are part of the token
        response = response.replace("<|end_of_sentence|>", "")

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
            "json": {
                "strict": bool,
                "allow_code_fence": bool,
                "schema": {
                    "keys": {"key1": "type", ...},
                    "no_extra_keys": bool
                }
            }
        }
        """
        # Parse constraints
        json_constraints = constraints.get("json", {})
        strict = json_constraints.get("strict", False)
        allow_code_fence = json_constraints.get("allow_code_fence", True)
        require_single_line = json_constraints.get("require_single_line", False)
        schema = json_constraints.get("schema")

        # 1. Extract JSON candidate
        json_str = response.strip()
        
        # Check single line constraint BEFORE extraction if possible, or after?
        # Usually better to check the raw response or the extracted part.
        # Let's check the extracted part to be fair (ignoring surrounding text newlines).
        
        # Regex to find code blocks
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, json_str, re.DOTALL)
        
        if match:
            if not allow_code_fence:
                return False, "Markdown code fences are not allowed"
            json_str = match.group(1)
        else:
            # Fallback: try to find the first '{' and last '}'
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1:
                extracted = json_str[start:end+1]
                # If strict is True, we might want to check if there is extra content, 
                # but for now let's focus on the JSON validity and keys.
                json_str = extracted
            else:
                # If no braces found, it's definitely not a JSON object
                return False, "No JSON object found (missing braces)"

        # Check single line constraint
        if require_single_line:
            if '\n' in json_str:
                return False, "JSON must be on a single line (no newlines allowed)"

        # 2. Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Extract context around the error
            idx = e.pos
            start = max(0, idx - 20)
            end = min(len(json_str), idx + 20)
            snippet = json_str[start:end]
            return False, f"Invalid JSON syntax: {e.msg}. Context: ...{repr(snippet)}..."
        
        if not isinstance(data, dict):
             return False, "JSON must be an object (dict), not a list or primitive"

        # 3. Check Keys
        required_keys = []
        no_extra_keys = False
        
        if schema:
            keys_dict = schema.get("keys", {})
            if keys_dict:
                required_keys = list(keys_dict.keys())
            no_extra_keys = schema.get("no_extra_keys", False)
        else:
            # Fallback to top-level required_keys if schema is not present
            required_keys = constraints.get("required_keys", [])

        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            return False, f"Missing required keys: {', '.join(missing_keys)}"
            
        if no_extra_keys:
            extra_keys = [k for k in data if k not in required_keys]
            if extra_keys:
                return False, f"Found extra keys: {', '.join(extra_keys)}"
            
        return True, None

def get_checker(task_type: str) -> BaseChecker:
    if task_type == "forbidden_words" or task_type == "taboo":
        return TabooChecker()
    elif task_type == "json_format" or task_type == "json":
        return JsonChecker()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
