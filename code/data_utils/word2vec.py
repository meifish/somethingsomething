import gensim
import os
import h5py
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path


class Word2Vec:
    def __init__(self, path_word2vec):
        #'word2vec/GoogleNews-vectors-negative300.bin'
        self.path_word2vec = path_word2vec

        #Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.path_word2vec, binary=True)
        self.original_keys = list(self.model.vocab.keys())
        self.upper_keys = [str.upper(x) for x in self.original_keys]


    def fix_typo(self, word):
        typo_path = Path.cwd() / "code/data_utils/missing.txt"
        typo = open(typo_path,'r')
        typo_dict = dict([(pairs.strip().split(',')[0], pairs.strip().split(',')[1]) for pairs in typo.readlines()])
        typo.close()

        if word in typo_dict:
            word = typo_dict[word]
        
        return word

    def uni_word(self, word):
        word = str.upper(word)
        word = word.replace(' ', '_')
        return word


    def return_weight(self, word):
        try:
            index = self.upper_keys.index(self.uni_word(word))
        except:
            return -1
        
        return self.model[self.original_keys[index]]


    def get_vec(self, word):

        found = False        
        token = self.fix_typo(word) 
        token_len = len(token.split(' '))

        while (token_len > 0):
            
            # Found
            if self.uni_word(token) in self.upper_keys:
                weight = self.return_weight(token)
                
                if len(weight) == 300:
                    return self.return_weight(token)   # numpy nd array

            # Not-Found
            else:
                if token_len == 1:
                    return self.return_weight("something")
           
                # Shorten the token and find again (Noun root is right oriented)
                token = ' '.join(token.split(' ')[1:])
                token_len -= 1


def main():

    annotation_file = '/media/meiyu/0C9255199255091A/code/somethingsomething/annotation.json'
    word2vec = Word2Vec('word2vec/GoogleNews-vectors-negative300.bin')
    
    found_words = {}
    missing_words = {}

    with open(annotation_file, 'r') as anno:
        annotation = json.load(anno)
        for video, frames in annotation.items():  # 'str' : 'list'
            print(f'video: {video}')
            objects = [label['category'] for f in frames for label in f['labels']]
            objects = list(set(objects))
            
            for word in objects:
                word = fix_typo(word)

                found = False
                token = word
                token_len = len(token.split(' '))
                print(f"word: {word}")
                while (token_len > 0):
                    print(f"token: {token}")
                    if (token not in found_words) and (token not in missing_words):
                        # Found
                        if word2vec.uni_word(token) in word2vec.upper_keys:
                            found_words[token] = word2vec.get_vec(token).tolist()
                            found = True
                            break
                        # Not-Found
                        else:
                            if token_len == 1:
                                missing_words[word] = video
                                break           
                            token = ' '.join(token.split(' ')[1:])
                            token_len -= 1
                    else:
                        found = True
                        break

                if found == False:
                    missing_words[word] = video
        
        with open('found_words.txt', 'w') as outfile:
            json.dump(found_words, outfile)

        with open('missing_words.txt', 'w') as outfile:
            json.dump(missing_words, outfile)
        

        print(f"missing word: {len(missing_words)}")
        print(missing_words)
        print(f"found word: {len(found_words)}")

    # print(word2vec.get_vec('hands'))
    # print([x for x in word2vec.upper_keys if 'SHIRT' in x])


if __name__ == '__main__':
    main()
