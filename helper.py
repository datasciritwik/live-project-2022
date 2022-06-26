def name_change(text):
    if text == 'Medquest Diagnostics Center' or text == 'Medquest Diagnostics':
        return 'Medquest Diagnostics Center'
    elif text == 'Pronto Diagnostics' or text == 'Pronto Diagnostics Center':
        return 'Pronto Diagnostics Center'
    elif text == 'Vijaya Diagonstic Center' or text == 'Vijaya Diagnostic Center':
        return 'Vijaya Diagnostic Center'
    elif text == 'Viva Diagnostic' or text == 'Vivaa Diagnostic Center':
        return 'Vivaa Diagnostic Center'
    else:
        return text