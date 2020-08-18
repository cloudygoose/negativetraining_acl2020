mkdir process_lm
#python process_lm.py process_oneline/train/train.txt > process_lm/train.txt
mkdir -p process_lm/test_25
mkdir -p process_lm/valid_25
python process_lm.py process_oneline/test_25/test.txt > process_lm/test_25/test.txt
python process_lm.py process_oneline/valid_25/valid.txt > process_lm/valid_25/valid.txt
