import numpy as np

def conll2span(label):
        stack = []
        label = np.array(label)
        state = {}
        labels = []
        for idx, tag in enumerate(label):
            state['Nb'] = tag.split('-')[0]
            state['Nt'] = tag.split('-')[-1]
            if state['Nb'] == "S":
                labels.append((idx,idx+1,state['Nt']))
            elif state['Nb'] == "B": 
                stack.append((idx, state['Nt']))
            elif state['Nb'] == "E":
                if state['Nt'] == stack[-1][1]:
                    temp_tag = stack.pop()
                    labels.append((temp_tag[0], idx+1, state['Nt']))
                else:
                    raise("Error :: Unbalanced") 
        if len(stack) != 0: 
            print("tag", tag)
            print("\nstack", stack)
            print('\nLabel', labels)
            raise "Error :: Unbalanced"
        return labels