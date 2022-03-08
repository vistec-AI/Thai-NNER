from seqeval.scheme import IOBES, IOB2
from seqeval.metrics import classification_report


def nne_conll_eval(batch_output, batch_targets, batch_mask, boundary_type,ids2tag=None, info=True):
    boundary_type = set(boundary_type)
    if boundary_type == set("BIOES"): scheme = IOBES
    elif boundary_type == set("BIO"): scheme = IOB2

    mask = []; output = []; targets = []
    for layer in range(len(batch_targets)):
        mask.extend(batch_mask.view(-1).cpu().numpy())
        targets.extend(batch_targets[layer].view(-1).cpu().numpy())
        output.extend(batch_output[layer].argmax(axis=1).view(-1).cpu().numpy())

    results = {'targets':[], 'predictions':[]}
    for index in range(len(targets)):    
        if mask[index]==1:
            temp_targets = targets[index]
            temp_output = output[index]
            if temp_output not in ids2tag:
                breakpoint()
            tag = ids2tag.get(temp_targets)
            pred = ids2tag.get(temp_output)
            results['targets'].append(tag)
            results['predictions'].append(pred)
    try:
        temp_out_results = classification_report(
            [results['targets']],
            [results['predictions']],
            mode='strict',
            scheme=scheme,
            digits=8)
    except:
        if set(results['targets'])=={'O'}:
            return 0
        else: breakpoint()
            
    if info: print(temp_out_results)
    temp_out_results = temp_out_results.split()
    assert temp_out_results[-18]=='micro' 
    f1 = float(temp_out_results[-14])
    return f1
    