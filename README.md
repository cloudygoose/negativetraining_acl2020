This is code for "Negative Training for Neural Dialogue Response Generation", which is to appear in ACL 2020.

The code uses pytorch 0.4, python 2.7, cuda 9.0, and nltk.

The Switchboard data is provided.

The main files are in seq2seq/ dir.

=== Train Baseline Models ===

Call lm_baseline.py or latent_baseline.py to train baseline models
Set EXP_ROOT at the beginning of the main files to set the place you want to save your models

To train them, for example:
python lm_baseline.py/latent_baseline.py "COMMAND='train';"

After training, use the test command to get PPL results:
python lm_baseline.py/latent_baseline.py "COMMAND='test';"

Note that the final checkpoint of baseline exps are going to be used for further steps

=== Negative Training for Malicious Responses ===

Simply call neg_mal_advtrain_seq2seq.py with default configuration.

Note that the malicious targets are provided in para_adv_lists/res_500.

You can play with training list augmentation with "ADV_TARGET_FN_TRAIN = '../para_adv_lists/res_500/train_500_cp1.txt';" E.g. You can change "cp1" to "cp2" or "cp5".

When ADV_I_LM_FLAG = False, it means o-sample-avg-hit is considered, when True, io-sample-avg-hit is considered.

=== Negative Training for Frequent Responses ===

Simply call neg_freq_posttrain_seq2seq.py with default configuration.

You can play with r_thres by setting "R_THRES=0.001", for example.

Tianxing He
cloudygoose@csail.mit.edu
