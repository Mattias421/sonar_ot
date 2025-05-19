source $EXP/sonar_ot/.venv/bin/activate
model=$1
lang=hin
lang_fleurs=hi_in
lang_bert=hi
script=Deva
refs=$DATA/fleurs/$lang_fleurs/
uv run emb_to_text.py results/$model/x1_pred/ --target-lang ${lang}_$script --output-prefix hyps
uv run emb_to_text.py results/$model/x1_pred/ --target-lang eng_Latn --output-prefix hyps_en
bert-score -r $refs/refs.txt -c results/$model/x1_pred/hyps.txt --lang $lang_bert
bert-score -r $refs/refs_en.txt -c results/$model/x1_pred/hyps_en.txt --lang en
sacrebleu $refs/refs.txt -i results/$model/x1_pred/hyps.txt --tokenize flores200
sacrebleu $refs/refs_en.txt -i results/$model/x1_pred/hyps_en.txt --tokenize flores200


