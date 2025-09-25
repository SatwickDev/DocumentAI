#!/usr/bin/env python3
"""
OCR Validation Engine - Lightning fast rule engine for post-OCR validation
Understands English validation rules and applies them to JSON data
"""

import json
import re
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any
import spacy
import pickle
from pathlib import Path

class ValidationEngine:
    def __init__(self, models_dir="models", cache_dir="cache"):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            sys.exit(1)
        self.compiled_rules = {}
        self.rule_cache_file = self.cache_dir / "compiled_rules.pkl"
        self._load_rule_cache()

    def _load_rule_cache(self):
        if self.rule_cache_file.exists():
            try:
                with open(self.rule_cache_file, 'rb') as f:
                    self.compiled_rules = pickle.load(f)
                print(f"Loaded {len(self.compiled_rules)} cached rules")
            except Exception as e:
                print(f"Warning: Could not load rule cache: {e}")
                self.compiled_rules = {}

    def _save_rule_cache(self):
        try:
            serializable_rules = {}
            for k, v in self.compiled_rules.items():
                new_rules = []
                for rule in v.get('rules', []):
                    r = dict(rule)
                    r.pop('validator', None)
                    new_rules.append(r)
                newv = dict(v)
                newv['rules'] = new_rules
                serializable_rules[k] = newv
            with open(self.rule_cache_file, 'wb') as f:
                pickle.dump(serializable_rules, f)
        except Exception as e:
            print(f"Warning: Could not save rule cache: {e}")

    @staticmethod
    def validate_greater_than(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        try:
            value = float(str(data[field]).replace(',', '').replace('$', '').replace('USD', '').replace('EUR', '').strip())
            if value > params['threshold']:
                return True, "Valid"
            return False, f"Value {value} is not greater than {params['threshold']}"
        except (ValueError, TypeError):
            return False, f"Invalid numeric value: {data[field]}"

    @staticmethod
    def validate_less_than(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        try:
            value = float(str(data[field]).replace(',', '').replace('$', '').replace('USD', '').replace('EUR', '').strip())
            if value < params['threshold']:
                return True, "Valid"
            return False, f"Value {value} is not less than {params['threshold']}"
        except (ValueError, TypeError):
            return False, f"Invalid numeric value: {data[field]}"

    @staticmethod
    def validate_between(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        try:
            value = float(str(data[field]).replace(',', '').replace('$', '').replace('USD', '').replace('EUR', '').strip())
            if params['min_val'] <= value <= params['max_val']:
                return True, "Valid"
            return False, f"Value {value} is not between {params['min_val']} and {params['max_val']}"
        except (ValueError, TypeError):
            return False, f"Invalid numeric value: {data[field]}"

    @staticmethod
    def validate_exact_match(field, params, data):
        expected = str(params['expected']).upper().strip()
        actual = str(data[field]).upper().strip()

        # Special handling for field 40E: treat "UCP" and "UCP LATEST VERSION" as equivalent
        if field.upper() == "40E":
            normalized_expected = expected.replace("LATEST VERSION", "").strip()
            normalized_actual = actual.replace("LATEST VERSION", "").strip()
            # Accept if either is exactly UCP, or both start with UCP
            if normalized_expected == "UCP" and normalized_actual == "UCP":
                return True, "Valid"
            if expected in ("UCP LATEST VERSION", "UCP") and actual in ("UCP LATEST VERSION", "UCP"):
                return True, "Valid"
            if normalized_expected == "UCP" and normalized_actual.startswith("UCP"):
                return True, "Valid"
            if normalized_actual == "UCP" and normalized_expected.startswith("UCP"):
                return True, "Valid"

        # Generic fallback: accept if actual == expected (full phrase)
        if actual == expected:
            return True, "Valid"
        # Accept if actual matches the first word/segment of expected
        expected_first = expected.split()[0]
        if actual == expected_first:
            return True, "Valid"
        return False, f"Expected '{expected}', got '{actual}'"

    @staticmethod
    def validate_contains(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        if params['required_text'].upper() in str(data[field]).upper():
            return True, "Valid"
        return False, f"Must contain '{params['required_text']}'"

    @staticmethod
    def validate_date_format(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        date_str = str(data[field]).strip()
        if params['format'] == 'DDMMYY':
            pattern = r'^\d{6}$'
            if re.match(pattern, date_str):
                try:
                    day = int(date_str[:2])
                    month = int(date_str[2:4])
                    year = int('20' + date_str[4:6])
                    datetime(year, month, day)
                    return True, "Valid date format"
                except ValueError:
                    return False, f"Invalid date: {date_str}"
        elif params['format'] == 'YYMMDD':
            pattern = r'^\d{6}$'
            if re.match(pattern, date_str):
                try:
                    year = int('20' + date_str[:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    datetime(year, month, day)
                    return True, "Valid date format"
                except ValueError:
                    return False, f"Invalid date: {date_str}"
        return False, f"Invalid date format: {date_str}"

    @staticmethod
    def validate_currency_format(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        value_str = str(data[field]).strip()
        currency_pattern = r'^([A-Z]{3})\s*([\d,.]+)$'
        m = re.match(currency_pattern, value_str)
        if m:
            amount_str = m.group(2).replace(',', '')
            try:
                amount = float(amount_str)
                if amount > 0:
                    return True, "Valid currency format"
                return False, "Amount must be greater than 0"
            except ValueError:
                return False, f"Invalid amount: {amount_str}"
        return False, f"Invalid currency format: {value_str}"

    @staticmethod
    def validate_max_length(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        value_str = str(data[field])
        if 'max_lines' in params:
            lines = value_str.split('\n')
            if len(lines) > params['max_lines']:
                return False, f"Too many lines: {len(lines)} > {params['max_lines']}"
            for line in lines:
                if len(line) > params['max_chars_per_line']:
                    return False, f"Line too long: {len(line)} > {params['max_chars_per_line']}"
        else:
            if len(value_str) > params['max_length']:
                return False, f"Too long: {len(value_str)} > {params['max_length']}"
        return True, "Valid length"

    @staticmethod
    def validate_required(field, params, data):
        if field not in data or data[field] is None or str(data[field]).strip() == '':
            return False, f"Field {field} is required"
        return True, "Valid"

    @staticmethod
    def validate_not_empty(field, params, data):
        if field not in data or not str(data[field]).strip():
            return False, f"Field {field} cannot be empty"
        return True, "Valid"

    @staticmethod
    def validate_unique(field, params, data):
        if field not in data or not str(data[field]).strip():
            return False, f"Field {field} must have a unique value"
        return True, "Valid (unique check skipped)"

    @staticmethod
    def validate_date_comparison(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        compare_field = params['compare_with']
        if compare_field not in data:
            return False, f"Comparison field {compare_field} not found"
        try:
            date1_str = str(data[field]).strip()
            date2_str = str(data[compare_field]).strip()
            def parse_date(date_str):
                if re.match(r'^\d{6}$', date_str):
                    try:
                        year = int('20' + date_str[:2])
                        month = int(date_str[2:4])
                        day = int(date_str[4:6])
                        return datetime(year, month, day)
                    except:
                        day = int(date_str[:2])
                        month = int(date_str[2:4])
                        year = int('20' + date_str[4:6])
                        return datetime(year, month, day)
                raise ValueError
            date1 = parse_date(date1_str)
            date2 = parse_date(date2_str)
            if params['operator'] == '>=' and date1 >= date2:
                return True, "Valid date comparison"
            elif params['operator'] == '>' and date1 > date2:
                return True, "Valid date comparison"
            return False, f"Date {field} ({date1_str}) must be >= {compare_field} ({date2_str})"
        except (ValueError, IndexError):
            return False, f"Invalid date format in comparison"

    @staticmethod
    def validate_multiple_choice(field, params, data):
        if field not in data:
            return False, f"Field {field} not found"
        value = str(data[field]).upper().strip()
        if value in params['choices']:
            return True, "Valid choice"
        return False, f"Must be one of: {', '.join(params['choices'])}"

    def parse_english_rule(self, rule_text: str, field_tag: str) -> Dict[str, Any]:
        """
        Parse English validation rule using NLP and pattern matching.
        For "should be" rules, extract the phrase after the keyword, up to a delimiter or EOL.
        """
        rule_info = {
            'original_text': rule_text,
            'field': field_tag,
            'type': 'unknown',
            'parameters': {}
        }
        text_lower = rule_text.lower()

        # 1. Unique + max length
        if 'unique' in text_lower and ('max' in text_lower or 'maximum' in text_lower):
            m = re.search(r'max(?:imum)?\s*(\d+)', text_lower)
            if m:
                rule_info['type'] = 'max_length'
                rule_info['parameters']['max_length'] = int(m.group(1))
                rule_info['parameters']['unique'] = True
                return rule_info

        # 2. Max length (with 35x4 or similar)
        m = re.search(r'max(?:imum)?\s*(\d+)\s*[x×]\s*(\d+)', text_lower)
        if m:
            rule_info['type'] = 'max_length'
            rule_info['parameters']['max_chars_per_line'] = int(m.group(1))
            rule_info['parameters']['max_lines'] = int(m.group(2))
            return rule_info
        m = re.search(r'max(?:imum)?\s*(\d+)\s*(?:characters|chars)?', text_lower)
        if m:
            rule_info['type'] = 'max_length'
            rule_info['parameters']['max_length'] = int(m.group(1))
            return rule_info

        # 3. Date format
        if 'yymmdd' in text_lower:
            rule_info['type'] = 'date_format'
            rule_info['parameters']['format'] = 'YYMMDD'
            return rule_info
        if 'ddmmyy' in text_lower:
            rule_info['type'] = 'date_format'
            rule_info['parameters']['format'] = 'DDMMYY'
            return rule_info

        # 4. Currency format
        if 'iso 4217' in text_lower:
            rule_info['type'] = 'currency_format'
            return rule_info

        # 5. Date comparison (cross-field)
        if ('expiry' in text_lower or 'expire' in text_lower or 'expiry date' in text_lower) and (
            'greater than or equal' in text_lower or 'greater than' in text_lower or 'after' in text_lower or '≥' in text_lower):
            if 'issue date' in text_lower:
                compare_with = '31C'
            else:
                m = re.search(r'compare[d]? with (\w+)', text_lower)
                compare_with = m.group(1).upper() if m else '31C'
            rule_info['type'] = 'date_comparison'
            rule_info['parameters']['compare_with'] = compare_with
            rule_info['parameters']['operator'] = '>='
            return rule_info

        # 6. Contains
        if any(word in text_lower for word in ['contain', 'includes', 'must have']):
            m = re.search(r'(contain|includes|must have)\s*([A-Za-z0-9\s]+)', text_lower)
            if m:
                rule_info['type'] = 'contains'
                rule_info['parameters']['required_text'] = m.group(2).strip().upper()
                return rule_info

        # 7. Multiple choice (should be X or Y or Z)
        if re.search(r'should be [A-Za-z\s]+ or [A-Za-z\s]+', text_lower) or re.search(r'must be [A-Za-z\s]+ or [A-Za-z\s]+', text_lower):
            m = re.search(r'(should|must) be ([\w\s]+)', text_lower)
            if m:
                choices_part = m.group(2)
                choices = [c.strip().upper() for c in re.split(r'\s+or\s+', choices_part)]
                rule_info['type'] = 'multiple_choice'
                rule_info['parameters']['choices'] = choices
                return rule_info

        # 8. Greater/Less/Between
        if 'greater than' in text_lower or 'more than' in text_lower or 'above' in text_lower or 'exceeds' in text_lower or 'at least' in text_lower or 'not less than' in text_lower:
            numbers = [int(s) for s in re.findall(r'\d+', text_lower)]
            if numbers:
                rule_info['type'] = 'greater_than'
                rule_info['parameters']['threshold'] = float(numbers[0])
                return rule_info
        if 'less than' in text_lower or 'below' in text_lower or 'under' in text_lower or 'at most' in text_lower or 'not more than' in text_lower:
            numbers = [int(s) for s in re.findall(r'\d+', text_lower)]
            if numbers:
                rule_info['type'] = 'less_than'
                rule_info['parameters']['threshold'] = float(numbers[0])
                return rule_info
        if 'between' in text_lower or 'range' in text_lower or 'from' in text_lower:
            numbers = [int(s) for s in re.findall(r'\d+', text_lower)]
            if len(numbers) >= 2:
                rule_info['type'] = 'between'
                rule_info['parameters']['min_val'] = float(numbers[0])
                rule_info['parameters']['max_val'] = float(numbers[1])
                return rule_info

        # 9. Exact match (should always be, must be, equals, etc)
        # Extract phrase after "should be" up to for/if/comma/dot/eol
        m = re.search(
            r'(?:should always be|must always be|should be|must be|equals|equal to|is exactly|is always|is)\s*([A-Za-z0-9\/\-\s]+?)(?=\s+(for|if)\b|,|\.|$)',
            text_lower
        )
        if m:
            expected = m.group(1).strip().upper()
            rule_info['type'] = 'exact_match'
            rule_info['parameters']['expected'] = expected
            return rule_info

        return rule_info

    def compile_rule(self, rule_info: Dict[str, Any]) -> callable:
        field = rule_info['field']
        rule_type = rule_info['type']
        params = rule_info['parameters']
        validators = {
            'greater_than': self.validate_greater_than,
            'less_than': self.validate_less_than,
            'between': self.validate_between,
            'exact_match': self.validate_exact_match,
            'contains': self.validate_contains,
            'date_format': self.validate_date_format,
            'currency_format': self.validate_currency_format,
            'max_length': self.validate_max_length,
            'required': self.validate_required,
            'not_empty': self.validate_not_empty,
            'unique': self.validate_unique,
            'date_comparison': self.validate_date_comparison,
            'multiple_choice': self.validate_multiple_choice
        }
        def validator(data):
            if rule_type in validators:
                return validators[rule_type](field, params, data)
            else:
                return False, f"Unknown rule type: {rule_type}"
        return validator

    def load_config(self, config_path: str, doc_type: str = "LC") -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_rules = {}
        cache_key = f"{config_path}_{os.path.getmtime(config_path)}"
        if cache_key in self.compiled_rules:
            print(f"Using cached rules for {doc_type}")
            new_rules = []
            for rule in self.compiled_rules[cache_key]['rules']:
                rule_info = rule['rule_info']
                validator_func = self.compile_rule(rule_info)
                new_rule = dict(rule)
                new_rule['validator'] = validator_func
                new_rules.append(new_rule)
            config_rules[cache_key] = {
                'doc_type': doc_type,
                'rules': new_rules,
                'config_path': config_path,
                'total_rules': len(new_rules)
            }
            return config_rules[cache_key]
        print(f"Parsing new rules for {doc_type}...")
        start_time = time.time()
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        parsed_rules = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                print(f"Warning: Invalid format in line {line_num}")
                continue
            field_tag, rule_text = line.split(':', 1)
            field_tag = field_tag.strip()
            rule_text = rule_text.strip()
            try:
                rule_info = self.parse_english_rule(rule_text, field_tag)
                validator_func = self.compile_rule(rule_info)
                parsed_rules.append({
                    'field': field_tag,
                    'rule_info': rule_info,
                    'validator': validator_func,
                    'original_text': rule_text
                })
            except Exception as e:
                print(f"Error parsing rule on line {line_num}: {e}")
                continue
        config_rules[cache_key] = {
            'doc_type': doc_type,
            'rules': parsed_rules,
            'config_path': config_path,
            'total_rules': len(parsed_rules)
        }
        self.compiled_rules[cache_key] = config_rules[cache_key]
        self._save_rule_cache()
        parse_time = time.time() - start_time
        print(f"Parsed {len(parsed_rules)} rules in {parse_time:.3f}s")
        return config_rules[cache_key]

    def validate_json(self, json_data: Dict[str, Any], config_rules: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        results = []
        total_passed = 0
        total_failed = 0
        for rule in config_rules['rules']:
            field = rule['field']
            validator = rule['validator']
            rule_info = rule['rule_info']
            try:
                is_valid, message = validator(json_data)
                result = {
                    'field': field,
                    'rule_type': rule_info['type'],
                    'original_rule': rule['original_text'],
                    'valid': is_valid,
                    'message': message,
                    'value': json_data.get(field, 'NOT_FOUND')
                }
                if is_valid:
                    total_passed += 1
                else:
                    total_failed += 1
                results.append(result)
            except Exception as e:
                result = {
                    'field': field,
                    'rule_type': rule_info['type'],
                    'original_rule': rule['original_text'],
                    'valid': False,
                    'message': f"Validation error: {str(e)}",
                    'value': json_data.get(field, 'NOT_FOUND')
                }
                total_failed += 1
                results.append(result)
        validation_time = time.time() - start_time
        return {
            'overall_valid': total_failed == 0,
            'document_type': config_rules['doc_type'],
            'total_rules': len(results),
            'passed': total_passed,
            'failed': total_failed,
            'validation_time_ms': round(validation_time * 1000, 2),
            'results': results,
            'json_data': json_data
        }

    def validate_document(self, config_path: str, json_data: Dict[str, Any], doc_type: str = "LC") -> Dict[str, Any]:
        try:
            config_rules = self.load_config(config_path, doc_type)
            validation_results = self.validate_json(json_data, config_rules)
            return validation_results
        except Exception as e:
            return {
                'overall_valid': False,
                'error': str(e),
                'document_type': doc_type,
                'validation_time_ms': 0,
                'results': []
            }

    def print_results(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("OCR VALIDATION RESULTS")
        print("="*80)
        if 'error' in results:
            print(f"❌ ERROR: {results['error']}")
            return
        status = "✅ VALID" if results['overall_valid'] else "❌ INVALID"
        print(f"Document Type: {results['document_type']}")
        print(f"Overall Status: {status}")
        print(f"Rules Processed: {results['total_rules']}")
        print(f"Passed: {results['passed']} | Failed: {results['failed']}")
        print(f"Validation Time: {results['validation_time_ms']}ms")
        print()
        passed_results = [r for r in results['results'] if r['valid']]
        failed_results = [r for r in results['results'] if not r['valid']]
        if failed_results:
            print("❌ FAILED VALIDATIONS:")
            print("-" * 60)
            for result in failed_results:
                print(f"Field: {result['field']}")
                print(f"Rule: {result['original_rule']}")
                print(f"Value: {result['value']}")
                print(f"Error: {result['message']}")
                print()
        if passed_results:
            print("✅ PASSED VALIDATIONS:")
            print("-" * 60)
            for result in passed_results:
                print(f"✓ {result['field']}: {result['original_rule']}")
        print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="OCR Validation Engine - Lightning fast English rule validation"
    )
    parser.add_argument("config", help="Configuration file with validation rules")
    parser.add_argument("json_file", help="JSON file to validate")
    parser.add_argument("--doc-type", default="LC", help="Document type (default: LC)")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    args = parser.parse_args()
    engine = ValidationEngine()
    try:
        with open(args.json_file, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
    results = engine.validate_document(args.config, json_data, args.doc_type)
    engine.print_results(results)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    sys.exit(0 if results['overall_valid'] else 1)

if __name__ == "__main__":
    main()