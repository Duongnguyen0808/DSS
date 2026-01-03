from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Template filter to get an item from a dictionary
    Usage: {{ mydict|get_item:key }}
    """
    if dictionary is None:
        return None
    return dictionary.get(key)

@register.filter
def format_number(value):
    """
    Format number with dot separator for thousands
    Usage: {{ number|format_number }}
    Example: 1000000 -> 1.000.000
    """
    try:
        value = float(value)
        return "{:,.0f}".format(value).replace(',', '.')
    except (ValueError, TypeError):
        return value
