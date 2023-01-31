# Compulsory Task 1 - Submitted by Bhavya Patteeswaran

import spacy
nlp = spacy.load('en_core_web_md')

# Code 1 - Similarities between cat, monkey and banana using 'en_core_web_md' language model
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(f"{word1} compare with {word2} : {word1.similarity(word2)}")
print(f"{word3} compare with {word2} : {word3.similarity(word2)}")
print(f"{word3} compare with {word1} : {word3.similarity(word1)}")

"""
output:
cat compare with monkey : 0.5929929675536907
banana compare with monkey : 0.4041501317354622
banana compare with cat : 0.22358825939615987

As above output, it clearly explain that 
- cat and monkey are animals so it score high compare to other similarities.
- monkey likes to eat banana than cat so similarity score is high compare to cat with banana.

"""

# Code 2 - with different example
tokens = nlp('chimpanzee lemon monkey dog orange')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

"""
output:
chimpanzee chimpanzee 1.0
chimpanzee lemon 0.09859715402126312
chimpanzee monkey 0.8372926115989685
chimpanzee banana 0.3217781186103821
chimpanzee orange 0.23419860005378723
lemon chimpanzee 0.09859715402126312
lemon lemon 1.0
lemon monkey 0.17996732890605927
lemon banana 0.6407583951950073
lemon orange 0.6341173052787781
monkey chimpanzee 0.8372926115989685
monkey lemon 0.17996732890605927
monkey monkey 1.0
monkey banana 0.404150128364563
monkey orange 0.24271909892559052
banana chimpanzee 0.3217781186103821
banana lemon 0.6407583951950073
banana monkey 0.404150128364563
banana banana 1.0
banana orange 0.5230288505554199
orange chimpanzee 0.23419860005378723
orange lemon 0.6341173052787781
orange monkey 0.24271909892559052
orange banana 0.5230288505554199
orange orange 1.0

As above output, it clearly explain that 
- same word will score 1.0 in similarity
- chimpanzee and monkey score 0.8 has both are animals and related to similar animal group.
- orange & lemon 0.6 shows belong to similar citrus fruit

"""

# Code 3 - Similarities between cat, monkey and banana using 'en_core_web_sm' language model
nlp1 = spacy.load('en_core_web_sm')
word4 = nlp1("cat")
word5 = nlp1("monkey")
word6 = nlp1("banana")

print(f"{word4} compare with {word5} : {word4.similarity(word5)}")
print(f"{word6} compare with {word5} : {word6.similarity(word5)}")
print(f"{word6} compare with {word4} : {word6.similarity(word4)}")

"""
Output:
Noticed with simpler language model result is less accurate score in terms of similarity factor in addition to the proof, 
it execute with the warning as:
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will 
be based on the tagger, parser and NER, which may not give useful similarity judgements. 
This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and 
only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models 
instead if available.
cat compare with monkey : 0.7371059361772669
banana compare with monkey : 0.7291608292298537
banana compare with cat : 0.6775488293064781

we can see the similar warning while trying on example python file attached with the task.

"""
