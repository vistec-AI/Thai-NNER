import json
import torch
import argparse
from tqdm import tqdm
import model.loss as module_loss
import model.model as module_arch
import model.metric as module_metric
from parse_config import ConfigParser
import data_loader.data_loaders as module_data

from model.eval import ClassEvaluator
from utils.conll2span import conll2span
from utils.span2json import span2json
from utils.correcting_labels import fix_labels, remove_incorrect_tag
PAD = '<pad>'


def get_dict_prediction(tokens, preds, attention_mask, ids2tag):
    temp_preds=[]
    for index in range(len(preds)):    
        if attention_mask[index]==1:
            Ptag = ids2tag.get(preds[index].item())
            temp_preds.append(Ptag)
            
    # Change BIO->BIESO to convert to Json format 
    temp_preds = remove_incorrect_tag(temp_preds, "BIOES")
    temp_preds = fix_labels(temp_preds, "BIOES")    
    temp_preds = conll2span(temp_preds)
    temp_preds = span2json(tokens, temp_preds)   
    return temp_preds


def _post_processing(results):
    post_processing = []
    for index in range(len(results)):
        predictions = []
        instance = results[index]
        for entity in instance['predictions']:
            skipt = False
            for FILTER in ["", "_", "<unk>", "/"]:
                if [FILTER]== entity['text']:
                    skipt=True; break
            if skipt: continue
            else: predictions.append(entity)

        post_processing.append({
            "sentence_id": instance['sentence_id'],
            "tokens":instance['tokens'],
            "entities": instance['entities'],
            "predictions": predictions})
    return post_processing


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader = data_loader.get_test()

    # build model architecturea
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    layers_train = config._config['trainer']['layers_train']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    results = []
    with torch.no_grad():
        for batch_idx, instance in enumerate(test_data_loader):
            input_ids = torch.tensor(instance['input_ids']).to(device)
            attention_mask = torch.tensor(instance['attention_mask']).to(device)
            batch_size = input_ids.shape[0]

            # Processing output
            output = model(input_ids, attention_mask)

            nested_lm_conll_ids = {l:None for l in range(len(layers_train))}
            for index, layer in enumerate(layers_train):
                temp_nested_lm_conll_ids = torch.tensor(instance['nested_lm_conll_ids'][layer])
                temp_nested_lm_conll_ids = temp_nested_lm_conll_ids.to(device)
                nested_lm_conll_ids[index]=temp_nested_lm_conll_ids
                loss+=criterion(output[index], temp_nested_lm_conll_ids)
            total_loss += loss.item() * batch_size

            predictions = {x:[] for x in range(batch_size)}
            lm_entities = {x:[] for x in range(batch_size)}
            for sent_ids in range(batch_size):
                for layer in range(len(output)):
                    predictions[sent_ids].append(output[layer][sent_ids].argmax(axis=0))
                    lm_entities[sent_ids].append(nested_lm_conll_ids[layer][sent_ids])

            for sent_ids in range(batch_size):
                tokens = instance['lm_tokens'][sent_ids]
                tokens = [w for w in tokens if w!=PAD]

                preds = []
                for index in range(len(layers_train)):
                    preds+=get_dict_prediction(
                            tokens, 
                            predictions[sent_ids][index], 
                            attention_mask[sent_ids], 
                            data_loader.ids2tag)

                entities_labels = []
                for index in range(len(layers_train)):
                    entities_labels+=get_dict_prediction(
                            tokens, 
                            lm_entities[sent_ids][index], 
                            attention_mask[sent_ids], 
                            data_loader.ids2tag)

                results.append({
                        'sentence_id': instance['sentence_id'][sent_ids],
                        'tokens': tokens,
                        'entities': entities_labels,
                        'predictions':preds})

                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric( 
                            output, 
                            nested_lm_conll_ids, 
                            attention_mask, 
                            data_loader.boundary_type,
                            info=False,
                            ids2tag=data_loader.ids2tag) * batch_size    

    n_samples = len(data_loader.test)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    logger.info(log)

    # Save predictions
    checkpoint_id = str(config.resume).split('/')[-2]
    path = f"outputs/preds_{config._config['name']}_{checkpoint_id}"

    # Save predictions
    with open(path+".json", 'w') as F:
        json.dump(results, F)
    print(f"Saved at: {path}")

    ## Can input both BIESO and BIO
    CE = ClassEvaluator()
    post_processing = _post_processing(results)
    json_results, conll_results = CE(post_processing)

    # Save conll
    with open(path+".conll", 'w') as Fconll:
        Fconll.writelines(f"\nCheckpoint: {config.resume}\n")
        Fconll.writelines(conll_results)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
