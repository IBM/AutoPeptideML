def format_numbers(obj):
    if isinstance(obj, dict):
        return {k: format_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_numbers(item) for item in obj]
    elif isinstance(obj, str):
        # Try to convert to int or float
        try:
            if '.' in obj or 'e' in obj.lower():
                return float(obj)
            else:
                return int(obj)
        except ValueError:
            return obj
    else:
        return obj
