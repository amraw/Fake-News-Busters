## Fake-News Busters
1. Requirements for running the code
   1. scikit-learn
   2. keras
   3. numpy
   4. tqdm
   5. tensorflow
   
2. Download the global vector file <a href="http://nlp.stanford.edu/data/glove.6B.zip"> glove.6B.zip </a> and put the glove.6B.100d.txt file in gloVe folder

3. For running the code: - <br/>
   python file_name body_truncation_length number_epochs
   <br/>
   Example:- <br/>
   
   1. python lstm_model_with_seperate_headline_body_encoding.py 75 10
   2. python lstm_model_headline_body_combined.py 75 10
   3. python lstm_model_with_global_features.py 75 10

4. Folders :-
   1. 
