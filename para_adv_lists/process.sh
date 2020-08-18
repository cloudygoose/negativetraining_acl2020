VOCABF=/data/sls/u/tianxing/projects/adversarial_seq2seq_201806/data/paranmt50m/res_5m/vocab.h50k 
mkdir res_500
#python process_ini.py --file=words_ori/mal_manual_ori_500.txt --vocab_file=$VOCABF --out_train_file=./res_500/train_500_cp1.txt --out_test_file=./res_500/test_500.txt --out_pair_file=./res_500/ori_pair.txt
echo "call python latent_baseline.py DATA_SET='paranmt_10m'; COMMAND='para_adv';"
python aug_train.py --train_file=./res_500/train_500_cp1.txt --para_file=./res_500/train_para_all.txt --out_train_file=./res_500/train_500_cp2.txt --aug_num=2
python aug_train.py --train_file=./res_500/train_500_cp1.txt --para_file=./res_500/train_para_all.txt --out_train_file=./res_500/train_500_cp5.txt --aug_num=5
