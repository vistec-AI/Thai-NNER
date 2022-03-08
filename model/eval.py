import pandas as pd
from tabulate import tabulate
from collections import defaultdict

HEAD = ["tag", "precision", "recall", "f1-score", 
        "preds_true", "num_preds", "num_labels"]

### NEW ###
groups=[
    ['person', 'title', 'firstname', 'middle', 'last', 
     'nickname', 'nicknametitle', 'namemod', 'psudoname', 'role'],

    ['location', 'continent', 'country' , 'state', 'city' , 'district', 
     'sub_district', 'province', 'roadname' , 'address', 
     'soi', 'latitude', 'longtitude' , 'postcode', 'ocean', 
     'island', 'mountian', 'river', 'space', 'restaurant', 
     'loc_others'],

    ['date', 'year', 'month', 'day', 'time', 
     'duration', 'periodic' , 'season', 'rel'],

    ['organisation', 'orgcorp', 'org_edu', 'org_political', 'org_religious', 'org_other', 
    'goverment', 'army', 'sports_team', 'media', 'hotel', 'museum', 
    'hospital', 'band', 'jargon', 'stock_exchange', 'index', 'fund'],

    ['norp', 'nationality', 'religion', 'norp_political', 'norp_others'],

    ['facility', 'airport', 'port', 'bridge', 'building', 'stadium', 'station', 'facility_other'],

    ['event', 'sports_event', 'concert', 'natural_disaster', 'war', 'event_others'],

    ['woa', 'book', 'film', 'song', 'tv_show', 'woa'],

    ['misc', 'animate', 'game', 'language', 'law', 'award', 'electronics', 
    'weapon', 'vehicle' , 'disease', 'god', 'sciname', 'food_ingredient',
     'product_food', 'product_drug', 'animal_species'],

    ['num', 'cardinal', 'mult', 'fold', 'money', 'energy', 'speed', 
        'distance', 'weight', 'quantity', 'percent', 'temperature', 'unit']
]

group1 = ["cardinal", "person", "firstname", "unit", "goverment", "title", "country", "last", "role", "month", "province", "day", "date", "year", "quantity", "org_political", "media", "org_other", "loc_others", "district"]
group2 = ['facility_other', 'org_edu','duration','orgcorp','law','time','nationality','rel','norp_political', 'money', 'city','sub_district','event_others','mult','norp_others','roadname','percent','army','disease']
group3 = ['religion', 'nickname', 'book', 'language', 'river',  'continent',  'restaurant',  'state',  'psudoname',  'address', 'electronics', 'weapon',  'hospital', 'natural_disaster', 'jargon',  'product_food', 'distance', 
          'island',  'building', 'fund', 'animal_species',  'nicknametitle',  'sciname', 'food_ingredient', 'tv_show', 'hotel',  'vehicle', 'org_religious',  'bridge',  'soi', 'periodic', 'airport', 'middle',  'station', 'namemod', 
          'song', 'mountian', 'film', 'weight',  'award',  'ocean',  'space',  'energy',  'product_drug', 'port', 'museum',  'god', 'woa', 'stadium', 'fold', 'sports_event', 'war', 'animate', 'band', 'season',  'stock_exchange',  'game', 
          'postcode', 'sports_team', 'temperature', 'index',  'longtitude',  'latitude', 'concert', 'speed']


class ClassEvaluator():
    
    def __init__(self, check_tags=True):
        self.check_tags=check_tags

    def __call__(self, data, save_path=None):
        results = self._counting(data)
        report_results = []
        all_tags = set(results['counts_labels'].keys())
        all_tags.update(set(results['counts_predictions'].keys()))

        if self.check_tags:
            self._check_tag(all_tags)
        
        # Each classes 
        for tag in all_tags:
            labels_true = results['counts_labels'][tag]['true']
            num_labels  = results['counts_labels'][tag]['true']
            num_labels += results['counts_labels'][tag]['false']

            predictions_true = results['counts_predictions'][tag]['true']
            num_predictions = results['counts_predictions'][tag]['true']
            num_predictions += results['counts_predictions'][tag]['false']

            precision, recall, f1 = self.get_f1(labels_true=labels_true, 
                                                num_labels=num_labels,
                                                predictions_true=predictions_true, 
                                                num_predictions=num_predictions)

            report_results.append(
                {'tag': tag,
                 'precision': precision,
                 'recall': recall,
                 'f1-score': f1,
                 'predictions_true': predictions_true,
                 'num_predictions': num_predictions,
                 'num_labels': num_labels
                })

        # Total
        labels_true = results['labels_true']
        num_labels = results['num_labels']
        predictions_true = results['predictions_true']
        num_predictions = results['num_predictions']
        
        precision, recall, f1 = self.get_f1(
                        labels_true=labels_true, 
                        num_labels=num_labels,
                        predictions_true=predictions_true, 
                        num_predictions=num_predictions)
    
        report_results.append(
                {'tag': 'total',
                 'precision': precision,
                 'recall': recall,
                 'f1-score': f1,
                 'predictions_true': predictions_true,
                 'num_predictions': num_predictions,
                 'num_labels': num_labels
                })
        
        report_results = sorted(report_results, 
                    key=lambda x: -x['f1-score'])

        # For ccalculate f1-score
        print("Calculate F1-score based on.")
        print(f"labels_true: {results['labels_true']}")
        print(f"num_labels: {results['num_labels']}")
        print(f"predictions_true: {results['predictions_true']}")
        print(f"num_predictions: {results['num_predictions']}")
        print("\n<<< Results Evaluations >>>\n")
        
        show_results = []
        for index in range(len(report_results)):
            instance = report_results[index]
            tag = instance['tag']
            show_results.append([
                tag,
                instance['precision'],
                instance['recall'],
                instance['f1-score'],
                instance['predictions_true'],
                instance['num_predictions'],
                instance['num_labels'],
            ])
            
#             if (index+1)%20==0:
#                 show_results.append([])
#                 show_results.append(show_results[0])
        
        show_results = sorted(show_results, key=lambda x: -x[-1])

        show_results = tabulate([HEAD]+show_results)
        self.long_tail_eval(data)
        print(show_results)
        
        for group in groups:
            print(f"\n<<< {group[0]} >>>")
            print(tabulate([HEAD]+[x for x in show_results 
                                    if x[0] in group]))
    
        if save_path==None:
            return report_results, show_results
        
        else:            
            # Create and save dataframe
            columns=['tag', 'precision', 'recall', 'f1-score', 
                'predictions_true', 'num_predictions', 'num_labels']
            df = pd.DataFrame(report_results, columns=columns)
            df.to_csv(save_path, index=False) 
            print(f"save as : {save_path}")
            return report_results, show_results

    def _counting(self, data):

        counts_labels = defaultdict(lambda: {"true": 0, "false": 0})
        counts_predictions = defaultdict(lambda: {"true": 0, "false": 0})

        for index in range(len(data)):
            labels = data[index]['entities']
            labels = [(item['span'], item['entity_type']) 
                                    for item in labels]
            predictions = data[index]['predictions']
            predictions = [(item['span'], item['entity_type']) 
                                        for item in predictions]
            for index in range(len(labels)):
                item = labels[index]
                tag = item[1]
                if item in predictions:
                    counts_labels[tag]['true']+=1
                else:
                    counts_labels[tag]['false']+=1
            for index in range(len(predictions)):
                item = predictions[index]
                tag = item[1]
                if item in labels:
                    counts_predictions[tag]['true']+=1
                else:
                    counts_predictions[tag]['false']+=1

        labels_true = sum([item['true'] for _, item in counts_labels.items()])
        labels_false = sum([item['false'] for _, item in counts_labels.items()])
        num_labels = labels_true + labels_false

        predicts_true = sum([item['true'] for _, item in counts_predictions.items()])
        predicts_false = sum([item['false'] for _, item in counts_predictions.items()])
        num_predicts = predicts_true + predicts_false

        return {
                'num_labels': num_labels,
                'labels_true': labels_true,
                'labels_false': labels_false,
                'counts_labels': counts_labels,

                'num_predictions': num_predicts,
                'predictions_true': predicts_true,
                'predictions_false': predicts_false,
                'counts_predictions': counts_predictions }

    def get_total(self, data):
        results = self._counting(data)

        labels_true = results['labels_true']
        num_labels = results['num_labels']

        predictions_true = results['predictions_true']
        num_predictions = results['num_predictions']

        # Calculate totals f1-score
        precision, recall, f1 = self.get_f1(
                                    labels_true=labels_true,
                                    num_labels=num_labels,
                                    predictions_true=predictions_true,
                                    num_predictions=num_predictions)
        return precision, recall, f1

    def get_f1(self, labels_true=0, num_labels=0, 
                predictions_true=0, num_predictions=0):
        
        if labels_true==0 \
            or num_labels==0\
            or predictions_true==0\
            or num_predictions==0:
            return 0, 0, 0

        precision = float(predictions_true)/num_predictions
        recall = float(labels_true)/num_labels

        f1 = 2. / ( (1./precision) + (1./recall))

        precision=precision*100
        recall=recall*100
        f1=f1*100
        
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)

        return precision, recall, f1
    
    def long_tail_eval(self, data):
        report_results = [['group/n.class', 'precision', 'recall', 'f1', 
                'predictions_true', 'num_predictions', 'num_labels']]
                
        results = self._counting(data)
        all_tags = set(results['counts_labels'].keys())
        all_tags.update(set(results['counts_predictions'].keys()))

        if self.check_tags:
            self._check_tag(all_tags)

        for index, group in enumerate([group1, group2, group3]):
            labels_true, num_labels = 0, 0
            predictions_true, num_predictions = 0, 0
            for tag in group:
                labels_true += results['counts_labels'][tag]['true']
                num_labels += (results['counts_labels'][tag]['true'] 
                                +results['counts_labels'][tag]['false'])
                predictions_true += results['counts_predictions'][tag]['true']
                num_predictions += (results['counts_predictions'][tag]['true'] 
                                    +results['counts_predictions'][tag]['false'])
            precision, recall, f1 = self.get_f1(
                labels_true=labels_true, 
                num_labels=num_labels,
                predictions_true=predictions_true, 
                num_predictions=num_predictions)
            report_results.append([
                f"group {index}: {len(group)}", precision, recall, f1, 
                predictions_true, num_predictions, num_labels])
        print(tabulate(report_results))
    
    def _check_tag(self, all_tags):# Check groups tag and dataset tag
        all_base_tags_1 = set([x for item in groups for x in item])
        assert len([x for x in all_tags if x not in all_base_tags_1])==0

        long_tail_groups = [group1, group2, group3]
        all_base_tags_2 = set([x for item in long_tail_groups for x in item])
        assert len([x for x in all_tags if x not in all_base_tags_2])==0