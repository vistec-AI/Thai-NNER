from transformers import AutoTokenizer

class InputLM():
    def __init__(self, lm_path, max_length) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_path, model_max_length=max_length)

    def __call__(self, tokens, entities):
        tokens = tokens.copy()
        entities = entities.copy()
        input_ids, attention_mask, encode_dict = self._tokenizer_input(tokens)
        shifted_entities = self._shifted_entities_index(input_ids, entities, encode_dict)
        lm_tokens=[self.tokenizer.decode(w)for w in input_ids]
        
        return {
            'attention_mask':attention_mask,
            'input_ids':input_ids,
            'lm_tokens':lm_tokens,
            'lm_entities':shifted_entities,
            'encode_dict':encode_dict}

    def _tokenizer_input(self, tokens):
        max_length = self.max_length
        start_id = self.tokenizer.bos_token_id
        end_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        encode_dict = {}
        input_ids = [start_id]
        for index in range(len(tokens)):
            word = tokens[index]
            shifted = len(input_ids)
            ids = self.tokenizer.encode(word)
            ids = ids[1:-1]
            input_ids.extend(ids)
            encode_dict[index]=(shifted, shifted+len(ids))

        input_ids.append(end_id)  # Add end of word
        num_ids = len(input_ids)  # Create mask
        mask = [1]*num_ids
        mask+= [0]*(max_length-num_ids)
        assert len(mask)==max_length, 'Error create mask'
        input_ids+=[pad_id] * (max_length-num_ids)  # Add padding
        return input_ids, mask, encode_dict

    def _shifted_entities_index(self, input_ids, entities, encode_dict):
        shifted_entities = []
        for index in range(len(entities)):  # Shift labels index
            entity = entities[index]
            entity_type = entity['entity_type']
            start, end = entity['span']
            text = entity['text']
            (shifted_start, _) = encode_dict.get(start)  # shifting start, end
            (_, shifted_end) = encode_dict.get(end-1)
            decode_text = input_ids[shifted_start:shifted_end]
            decode_text = [self.tokenizer.decode(w) for w in decode_text]
            decode_text = "".join(decode_text)
            shifted_entities.append({
                'entity_type':entity_type,
                'span':[shifted_start, shifted_end],
                'text': decode_text})
        return shifted_entities


    @staticmethod
    def check_input_ids_and_mask(sample):
        temp = [['index', 'input_text', 'input_ids', 'mask']]
        for index in range(len(sample['input_ids'])):
            
            original_ids = sample['input_ids'][index]
            mask = sample['mask'][index]
            input_text = sample['input_text'][index]
            
            temp.append([index, input_text, original_ids, mask])