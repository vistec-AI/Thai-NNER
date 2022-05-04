def span2json(tokens, span_labels):
    results = []
    for item in span_labels:
        start, end, tag = item
        results.append({
            'text': tokens[start:end],
            'span':[start, end],
            'entity_type':tag})
    return results