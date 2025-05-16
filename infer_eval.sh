source $EXP/sonar_ot/.venv/bin/activate
model=$1
refs=$DATA/fleurs/ca_es/
uv run emb_to_text.py results/$model/x1_pred/ --target-lang eng_Latn --output-prefix hyps_en
uv run emb_to_text.py results/$model/x1_pred/ --target-lang cat_Latn --output-prefix hyps
bert-score -r $refs/refs.txt -c results/$model/x1_pred/hyps.txt --lang ca
bert-score -r $refs/refs_en.txt -c results/$model/x1_pred/hyps_en.txt --lang en
sacrebleu $refs/refs.txt -i results/$model/x1_pred/hyps.txt --tokenize flores200
sacrebleu $refs/refs_en.txt -i results/$model/x1_pred/hyps_en.txt --tokenize flores200


