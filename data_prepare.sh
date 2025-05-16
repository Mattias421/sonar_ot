cd $DATA/BembaSpeech/bem/

awk -F'\t' '!seen[$1]++' train.tsv > train.csv
sed -i 's/audio/ID/' train.csv
sed -i 's/\t/,/' train.csv
sed -i 's/.wav//' train.csv

awk -F'\t' '!seen[$1]++' dev.tsv > dev.csv
sed -i 's/audio/ID/' dev.csv
sed -i 's/\t/,/' dev.csv
sed -i 's/.wav//' dev.csv

awk -F'\t' '!seen[$1]++' test.tsv > test.csv
sed -i 's/audio/ID/' test.csv
sed -i 's/\t/,/' test.csv
sed -i 's/.wav//' test.csv

cd -
