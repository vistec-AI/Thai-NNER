def is_same_NE(NE1, NE2):
    state=0
    if NE1[0]=='B' and NE2[0]=='I': state=1
    if NE1[0]=='B' and NE2[0]=='E': state=1
    if NE1[0]=='I' and NE2[0]=='I': state=1
    if NE1[0]=='I' and NE2[0]=='E': state=1
    
    if state==1 and NE1[1:] == NE2[1:]:
        return True
    else:
        return False

def is_S(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==False:
        if prev2now == now2next:
            if now[0]!='O':
                return True
    else:
        return False

def is_B(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==False:
        if now2next==True:
            return True
    return False

def is_I(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==True:
        if now2next==True:
            return True
    return False

def is_E(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==True:
        if now2next==False:
            return True
    return False

def is_O(prev, now, next):
    if now[0]=='O':
        return True
    return False

def fix_labels(labels, tag_type, visualize=False):
    prev = "O"
    fixed_labels = []
    length = len(labels)
    for index in range(length):
        change_state=0
        now = labels[index]
        next = labels[index+1] if index < length-1 else "O"

        if is_O(prev, now, next):
            now="O"

        elif is_S(prev, now, next):
            if set(tag_type)==set("BIO"):
                now="B"+now[1:]
                if visualize:
                    print("B", end='')

            elif set(tag_type)==set("BIOES"):
                now="S"+now[1:]
                if visualize:
                    print("S", end='')
            else:
                raise "Error tag_type"

            change_state+=1
        
        if is_B(prev, now, next):
            now="B"+now[1:]
            change_state+=1
            if visualize:
                print("B", end='')
        
        if is_I(prev, now, next):
            now="I"+now[1:]
            change_state+=1
            if visualize:
                print("I", end='')

        if is_E(prev, now, next):
            if set(tag_type)==set("BIOES"):
                now="E"+now[1:]
                if visualize:
                    print("E", end='')

            elif set(tag_type)==set("BIO"):
                now="I"+now[1:]
                if visualize:
                    print("I", end='')
            else:
                raise "Error tag_type"
            change_state+=1
        prev=now
        fixed_labels.append(now)
        if change_state>1:
            print("Duble change")
            breakpoint()
        if visualize and change_state>=1 and labels[index]!=now:
            print(f"\t{labels[index]} -> {now}\t\t {labels[index-1], labels[index], next}")
    return fixed_labels

def remove_incorrect_tag(labels, tag_type):
    assert set(tag_type)==set("BIOES")
    prev = "O"
    count = 0
    state = False
    fixed_labels = []
    length = len(labels)
    START = 0; END = 0
    correct_entity = False
    
    for index in range(length):
        now = labels[index]
        next = labels[index+1] if index < length-1 else "O"
        prev2now = is_same_NE(prev, now)
        if prev2now: 
            pass
        else: 
            state = False
        if state:
            if now[0]=="E":
                END=index
                state = False
                correct_entity=True
        else:
            if now[0] in ['B']:
                count+= 1
                state = True
                START = index
        prev=now
        if correct_entity:
            fixed_labels.extend(list(range(START, END+1)))
            correct_entity=False 
        
        if now[0]=='S':
            fixed_labels.extend([index])  
    results = []
    for ids, tag in enumerate(labels):
        if ids in fixed_labels:
            results.append(tag)
        else:
            results.append("O")
    return results